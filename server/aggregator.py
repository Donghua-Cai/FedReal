import threading
import logging
import math
from typing import Dict, List, Tuple, Set, Optional

import torch

from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.model.create_model import create_model
from common.utils import select_clients

from .eval import evaluate

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

        self.server_eval_acc = []
        self.server_eval_loss = []

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
    def get_task(self, client_id: str):
        # 采样逻辑仍在内部加锁执行
        self._ensure_sampling()
        with self.lock:
            # 已经到总轮数：不再发模型
            if self.current_round >= self.cfg.total_rounds:
                return self.current_round, False, b""

            # 默认是否参与
            participate = client_id in self.selected_this_round and self.expected_updates > 0

            # 已经提交过更新的客户端，本轮不应再训练也不应再下发模型
            if client_id in self.completed_this_round:
                participate = False
                return self.current_round, False, b""

            # 仅在“本轮被采样参与且尚未完成”时下发模型
            model_bytes = self.global_bytes if participate else b""
            return self.current_round, participate, model_bytes

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
    def _aggregate_and_advance(self):
        # 将各客户端完整权重做样本数加权平均（仅浮点张量参与加权）
        state_dicts = []
        weights = []
        for b, n in self.received_updates.values():
            sd = bytes_to_state_dict(b)
            state_dicts.append(sd)
            weights.append(float(n))
        total = sum(weights)
        if total <= 0:
            weights = [1.0 for _ in weights]
            total = len(weights)

        # 以第一份的结构为模板
        template = state_dicts[0]
        agg = {}

        for k, v in template.items():
            if v.dtype.is_floating_point:
                # 浮点：用 float32 做加权求和更稳
                acc = torch.zeros_like(v, dtype=torch.float32)
                for sd, w in zip(state_dicts, weights):
                    acc += sd[k].to(torch.float32) * (w / total)
                # 回到原始 dtype（通常是 float32，本行等价但更稳健）
                agg[k] = acc.to(v.dtype)
            else:
                # 非浮点：直接取第一份（如 num_batches_tracked 等计数器）
                agg[k] = v.clone()

        self.model.load_state_dict(agg)
        self.global_bytes = state_dict_to_bytes(self.model.state_dict())

        # 评测
        if self.public_test_loader is not None:
            loss, acc = evaluate(self.model, self.public_test_loader, device=self.device)
            logger.info(f"[Round {self.current_round}] Global Eval — loss={loss:.4f}, acc={acc:.4f}")
            self.server_eval_acc.append(acc)
            self.server_eval_loss.append(loss)
        else:
            logger.debug(f"[Round {self.current_round}] Global Eval — skipped (no public_test_loader)")

        # 前进到下一轮
        self.current_round += 1
        self.selected_this_round = []
        self.expected_updates = 0
        self.received_updates.clear()
        self.completed_this_round.clear()

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