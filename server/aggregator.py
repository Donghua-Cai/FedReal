import threading
from typing import Dict, List, Tuple
import math
import torch

from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.model.create_model import create_model
from common.utils import select_clients


class Aggregator:
    def __init__(self, config, public_test_loader=None, device="cpu"):
        self.cfg = config
        self.device = torch.device(device)

        self.model = create_model(self.cfg.model_name, num_classes=10).to(self.device)
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
        self.require_full = True # 强同步 (必须所有num_clients都训练完才进入下一轮)

        # 暂存：public logits（按轮、按客户端、再按 chunk）
        self.public_logits_chunks = {}  # {(round, client_id): {chunk_id: (indices, logits_bytes, rows, num_classes)}}

    # —— 注册 ——
    def register(self, client_name: str) -> Tuple[str, int]:
        with self.lock:
            client_id = f"C{len(self.registered):03d}"
            self.registered.append(client_id)
            self.client_index[client_id] = len(self.client_index)
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

            print(f"[Server] ensure_sampling: registered={len(self.registered)} "
                  f"N={self.cfg.num_clients} sample_fraction={self.cfg.sample_fraction} round={self.current_round}")
            
            # 强同步：未满足配置的 num_clients 数量，不开轮（所有客户端都会拿到 participate=False）
            if self.require_full and len(self.registered) < self.cfg.num_clients:
                self.selected_this_round = []
                self.expected_updates = 0
                self.received_updates.clear()
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
                return

            self.selected_this_round = select_clients(self.registered, k, self.current_round, seed=self.cfg.seed)
            self.expected_updates = len(self.selected_this_round)
            self.received_updates.clear()

            # 采样成功后打印
            if self.selected_this_round:
                print(f"[Server] Round {self.current_round} sampling -> "
                    f"{self.selected_this_round}, expected_updates={self.expected_updates}")
            else:
                print(f"[Server] Round {self.current_round} not started (waiting)")

    # —— 获取任务 ——
    def get_task(self, client_id: str):
        # 采样逻辑仍在内部加锁执行
        self._ensure_sampling()
        with self.lock:
            if self.current_round >= self.cfg.total_rounds:
                return self.current_round, False, self.global_bytes
            # 如果本轮未开（强同步等待）或不在采样名单，直接不参与
            participate = client_id in self.selected_this_round
            # 已经上传过本轮更新的客户端，不再参与，避免重复训练
            if client_id in self.completed_this_round:
                participate = False
            return self.current_round, participate, self.global_bytes

    # —— 收到更新 ——
    def submit_update(self, client_id: str, round_id: int, local_bytes: bytes, num_samples: int):
        with self.lock:
            print(f"[Server] recv from {client_id} for round={round_id} "
                  f"(curr={self.current_round}), total_received={len(self.received_updates)+1}/{self.expected_updates}")
            # 仅接受当前轮更新
            if round_id != self.current_round:
                return False
            # 只收一次
            if client_id in self.received_updates:
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
            from .eval import evaluate
            loss, acc = evaluate(self.model, self.public_test_loader, device=self.device)
            print(f"[Server][Round {self.current_round}] Global Eval — loss={loss:.4f}, acc={acc:.4f}")

        # 前进到下一轮
        self.current_round += 1
        self.selected_this_round = []
        self.expected_updates = 0
        self.received_updates.clear()
        self.completed_this_round.clear()
    
    
    def accept_public_logits_payload(
        self,
        client_id: str,
        round_id: int,
        logits_bytes: bytes,
        indices: list[int] | None,
        num_classes: int,
        total_examples: int | None,
    ):
        key = (round_id, client_id)
        self.public_logits_payloads[key] = (indices, logits_bytes, num_classes, total_examples)
        # 现在先“接住”即可；你后面可在需要处解析成 numpy/tensor：
        #   import numpy as np
        #   rows = (len(logits_bytes) // 4) // num_classes
        #   logits = np.frombuffer(logits_bytes, dtype=np.float32).reshape(rows, num_classes)
        # 然后做蒸馏/一致性之类的处理。