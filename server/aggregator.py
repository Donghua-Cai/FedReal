import threading
import logging
import math
from typing import Dict, List, Tuple, Set, Optional

import torch

from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.model.create_model import create_model
from common.utils import select_clients

# 使用与 server_main.py 同一命名空间的子 logger，继承其 handler/level
logger = logging.getLogger("Server").getChild("Aggregator")


class Aggregator:
    def __init__(self, config, public_test_loader=None, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)

        self.model = create_model(self.cfg.model_name, num_classes=self.cfg.num_classes).to(self.device)
        self.global_bytes = state_dict_to_bytes(self.model.state_dict())

        self.public_test_loader = public_test_loader

        self.current_round = 0
        self.registered: List[str] = []
        self.client_index: Dict[str, int] = {}

        self.selected_this_round: List[str] = []
        self.expected_updates = 0
        self.received_updates: Dict[str, Tuple[bytes, int]] = {}
        self.lock = threading.Lock()

        self.completed_this_round: Set[str] = set()
        # 强同步：必须所有 num_clients 都训练完才进入下一轮
        self.require_full = True

        # 一次性公共 logits 暂存
        # {(round, client_id): (indices(list|None), logits_bytes(bytes), num_classes(int), total_examples(int|None))}
        self.public_logits_payloads: Dict[Tuple[int, str], Tuple[Optional[List[int]], bytes, int, Optional[int]]] = {}

    # —— 注册 ——
    def register(self, client_name: str) -> Tuple[str, int]:
        with self.lock:
            client_id = f"C{len(self.registered):03d}"
            self.registered.append(client_id)
            self.client_index[client_id] = len(self.client_index)
            logger.info(f"Registered {client_id} (index={self.client_index[client_id]})")
            return client_id, self.client_index[client_id]

    # —— 采样 ——
    def _ensure_sampling(self):
        with self.lock:
            # 终止条件：到达总轮数后，不再采样
            if self.current_round >= self.cfg.total_rounds:
                self.selected_this_round = []
                self.expected_updates = 0
                self.received_updates.clear()
                return

            # 如果本轮已经采样过，直接返回
            if self.expected_updates > 0 or self.selected_this_round:
                return

            logger.debug(
                f"ensure_sampling: registered={len(self.registered)} "
                f"N={self.cfg.num_clients} sample_fraction={self.cfg.sample_fraction} "
                f"round={self.current_round}"
            )

            # 强同步：未满足配置的 num_clients 数量，不开轮（所有客户端都会拿到 participate=False）
            if self.require_full and len(self.registered) < self.cfg.num_clients:
                self.selected_this_round = []
                self.expected_updates = 0
                self.received_updates.clear()
                logger.warning(
                    f"Round {self.current_round} not started: "
                    f"registered={len(self.registered)} < required={self.cfg.num_clients}"
                )
                return

            # 采样数量基于配置的 num_clients（而不是已注册数量）
            k_target = max(1, math.ceil(self.cfg.sample_fraction * self.cfg.num_clients))
            # 从已注册里采样，但数量不超过已注册
            k = min(k_target, len(self.registered))
            # 如果强同步，要求采样数必须等于目标（满编），否则不开始
            if self.require_full and k < k_target:
                self.selected_this_round = []
                self.expected_updates = 0
                self.received_updates.clear()
                logger.warning(
                    f"Round {self.current_round} not started: sampled={k} < target={k_target}"
                )
                return

            self.selected_this_round = select_clients(
                self.registered, k, self.current_round, seed=self.cfg.seed
            )
            self.expected_updates = len(self.selected_this_round)
            self.received_updates.clear()

            # 采样成功后打印
            if self.selected_this_round:
                logger.info(
                    f"Round {self.current_round} sampling -> "
                    f"{self.selected_this_round}, expected_updates={self.expected_updates}"
                )
            else:
                logger.warning(f"Round {self.current_round} not started (waiting)")

    # —— 获取任务 ——
    # server/aggregator.py
    def get_task(self, client_id: str) -> Tuple[int, int, bytes]:
        with self.lock:
            # 已完成：任何请求都返回 DONE
            if self.current_round >= self.cfg.total_rounds or self.phase == fed_pb2.PHASE_DONE:
                self.phase = fed_pb2.PHASE_DONE
                return self.current_round, fed_pb2.PHASE_DONE, b""

            # 强同步：未注册满 -> WAITING（只轮询不训练）
            if self.require_full and len(self.registered) < self.cfg.num_clients:
                logger.warning(
                    f"Round {self.current_round} waiting: "
                    f"registered={len(self.registered)} < required={self.cfg.num_clients}"
                )
                return self.current_round, fed_pb2.PHASE_WAITING, b""

            if self.phase == fed_pb2.PHASE_LOCAL_SUP:
                return self.current_round, fed_pb2.PHASE_LOCAL_SUP, b""

            if self.phase == fed_pb2.PHASE_CLIENT_KD:
                # 仅首次对该 client 下发 teacher logits
                if client_id not in self.teacher_sent_to and self.teacher_logits_bytes is not None:
                    self.teacher_sent_to.add(client_id)
                    return self.current_round, fed_pb2.PHASE_CLIENT_KD, self.teacher_logits_bytes
                return self.current_round, fed_pb2.PHASE_CLIENT_KD, b""

        # 兜底
        return self.current_round, fed_pb2.PHASE_WAITING, b""

    # —— 收到更新 ——
    def submit_update(self, client_id: str, round_id: int, local_bytes: bytes, num_samples: int):
        with self.lock:
            logger.info(
                f"recv from {client_id} for round={round_id} "
                f"(curr={self.current_round}), total_received={len(self.received_updates)+1}/{self.expected_updates}"
            )
            # 仅接受当前轮更新
            if round_id != self.current_round:
                logger.warning(
                    f"drop update from {client_id}: stale round={round_id} (curr={self.current_round})"
                )
                return False
            # 只收一次
            if client_id in self.received_updates:
                logger.debug(f"ignore duplicate update from {client_id}")
                return True

            self.received_updates[client_id] = (local_bytes, num_samples)
            self.completed_this_round.add(client_id)

            if len(self.received_updates) >= self.expected_updates:
                self._aggregate_and_advance()
            return True

    # —— 聚合 ——
    def _advance_round(self):
        # 清理本轮缓存
        self.received_logits.clear()
        self.kd_acks.clear()
        self.teacher_logits_bytes = None
        self.teacher_sent_to.clear()

        next_round = self.current_round + 1
        if next_round >= self.cfg.total_rounds:
            # 训练完成：进入 DONE，相位锁死
            logger.info(f"[Round {self.current_round}] Completed. All {self.cfg.total_rounds} rounds finished.")
            self.current_round = next_round
            self.phase = fed_pb2.PHASE_DONE
        else:
            logger.info(f"[Round {self.current_round}] Completed. Advancing to round {next_round}.")
            self.current_round = next_round
            self.phase = fed_pb2.PHASE_LOCAL_SUP

    # —— 接收一次性公共 logits —— 
    def accept_public_logits_payload(
        self,
        client_id: str,
        round_id: int,
        logits_bytes: bytes,
        indices: list[int] | None,
        num_classes: int,
        total_examples: int | None,
        local_train_samples: int | None,   # ✅ 新增
    ):
        key = (round_id, client_id)
        if not hasattr(self, "public_logits_payloads"):
            self.public_logits_payloads = {}
        self.public_logits_payloads[key] = {
            "indices": indices,
            "logits_bytes": logits_bytes,
            "num_classes": num_classes,
            "total_examples": total_examples,
            "local_train_samples": local_train_samples,  # ✅ 存起来
        }
        logger.info(
            f"[Server] cached public logits from {client_id} (round={round_id}), "
            f"bytes={len(logits_bytes)}, num_classes={num_classes}, "
            f"local_train_samples={local_train_samples}"
        )