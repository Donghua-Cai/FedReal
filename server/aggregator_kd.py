# server/aggregator_kd.py
import logging
import threading
from typing import Dict, List, Optional, Tuple, Set

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from proto import fed_pb2
from common.model.create_model import create_model  # 你的 create_model 应该支持 resnet50
from common.losses import KLDivergenceWithTemperature

from tqdm import tqdm

logger = logging.getLogger("Server").getChild("AggregatorKD")

class AggregatorKD:
    """
    强同步 KD 联邦
    - Phase A: clients 本地监督训练 + 上传公共集 logits（带 local_train_samples）
    - Phase B: server 聚合加权 logits -> 伪标签 -> server KD 训练；随后下发 teacher logits
              clients 在公共集 KD 训练完成后 ACK
    - 收齐 ACK -> 下一轮
    """
    def __init__(
        self,
        cfg,                             # FedConfig（含 kd 超参 / num_classes / public_examples）
        public_loader: DataLoader,       # 公共无标签 loader（shuffle=False）
        server_test_loader: Optional[DataLoader], # 可选：server 私有评测
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)

        # Server 端模型（异构：ResNet50）
        # 你的 create_model(name, num_classes) 需要支持 resnet50
        self.server_model: nn.Module = create_model("resnet50", num_classes=self.cfg.num_classes).to(self.device)

        self.public_loader = public_loader
        self.server_test_loader = server_test_loader

        # 训练状态
        self.current_round: int = 0
        self.phase: fed_pb2.Phase = fed_pb2.PHASE_LOCAL_SUP  # 初始等待客户端上传 logits

        # 注册&强同步
        self.registered: List[str] = []
        self.client_index: Dict[str, int] = {}
        self.require_full: bool = True

        # 本轮期望/接收
        self.expected_clients: int = self.cfg.num_clients
        self.received_logits: Dict[str, Tuple[torch.Tensor, int]] = {}  # client_id -> (logits[N,C], local_samples)
        self.kd_acks: Set[str] = set()

        # 下发 teacher logits 缓存（bytes） & 已下发标记
        self.teacher_logits_bytes: Optional[bytes] = None
        self.teacher_sent_to: Set[str] = set()

        self.lock = threading.Lock()

    # ---------- 注册 ----------
    def register(self, client_name: str) -> Tuple[str, int]:
        with self.lock:
            cid = f"C{len(self.registered):03d}"
            self.registered.append(cid)
            self.client_index[cid] = len(self.client_index)
            logger.info(f"Registered {cid} (index={self.client_index[cid]})")
            return cid, self.client_index[cid]

    # ---------- GetTask: 返回当前轮的 phase & 仅在 PHASE_CLIENT_KD 首次下发 teacher logits ----------
    # server/aggregator_kd.py
    def get_task(self, client_id: str) -> Tuple[int, int, bytes]:
        with self.lock:
            # --- 训练已完成：任何请求都返回 DONE ---
            if self.phase == fed_pb2.PHASE_DONE or self.current_round >= self.cfg.total_rounds:
                self.phase = fed_pb2.PHASE_DONE
                return self.current_round, fed_pb2.PHASE_DONE, b""

            # --- 强同步：未注册满，只返回 WAITING，不允许任何训练 ---
            if self.require_full and len(self.registered) < self.cfg.num_clients:
                logger.warning(
                    f"Round {self.current_round} waiting: "
                    f"registered={len(self.registered)} < required={self.cfg.num_clients}"
                )
                return self.current_round, fed_pb2.PHASE_WAITING, b""

            # --- 正常相位调度 ---
            if self.phase == fed_pb2.PHASE_LOCAL_SUP:
                # 客户端做本地监督训练并上传 logits
                return self.current_round, fed_pb2.PHASE_LOCAL_SUP, b""

            if self.phase == fed_pb2.PHASE_CLIENT_KD:
                # 客户端做 KD；仅首次对该 client 下发 teacher logits
                if client_id not in self.teacher_sent_to and self.teacher_logits_bytes is not None:
                    self.teacher_sent_to.add(client_id)
                    return self.current_round, fed_pb2.PHASE_CLIENT_KD, self.teacher_logits_bytes
                return self.current_round, fed_pb2.PHASE_CLIENT_KD, b""

            # 兜底（不应到达）
            return self.current_round, fed_pb2.PHASE_WAITING, b""

    # ---------- 接收客户端 logits ----------
    def accept_client_logits(
        self,
        client_id: str,
        round_id: int,
        logits_bytes: bytes,
        num_classes: int,
        total_examples: int,
        local_train_samples: int,
    ):
        with self.lock:
            if round_id != self.current_round:
                logger.warning(f"Drop logits from {client_id}: stale round={round_id} (curr={self.current_round})")
                return False

            # 反序列化
            import numpy as np
            arr = np.frombuffer(logits_bytes, dtype=np.float32).copy()
            if arr.size != (total_examples * num_classes):
                logger.error(
                    f"Bad logits size from {client_id}: got {arr.size}, expect {total_examples * num_classes}"
                )
                return False
            tensor = torch.from_numpy(arr).view(total_examples, num_classes)

            self.received_logits[client_id] = (tensor, int(local_train_samples))
            logger.info(
                f"[AggregatorKD] recv logits from {client_id} for round={round_id} "
                f"received={len(self.received_logits)}/{self.expected_clients}"
            )

            # 收齐，进入 Server 训练
            if len(self.received_logits) >= self.expected_clients:
                self._server_train_and_prepare_teacher_logits()
                # 进入 KD 阶段，等待 ACK
                self.phase = fed_pb2.PHASE_CLIENT_KD
                self.kd_acks.clear()
                self.teacher_sent_to.clear()
                logger.info(f"[Round {self.current_round}] Enter PHASE_CLIENT_KD (teacher logits ready).")
            return True

    # ---------- Server 端：聚合 + 伪标签 + KD 训练 + 生成 teacher logits ----------
    def _server_train_and_prepare_teacher_logits(self):
        # 1) 聚合（按 local_train_samples 加权）
        total_w = sum(w for _, w in self.received_logits.values())
        if total_w <= 0:
            total_w = len(self.received_logits)
        weights = {cid: w / total_w for cid, (_, w) in self.received_logits.items()}

        # 假定所有 [N,C] 一致
        any_tensor = next(iter(self.received_logits.values()))[0]
        N, C = any_tensor.shape

        agg_logits = torch.zeros_like(any_tensor, dtype=torch.float32)
        for cid, (logits, _) in self.received_logits.items():
            agg_logits += logits.to(torch.float32) * weights[cid]

        # 2) 伪标签
        pseudo = agg_logits.argmax(dim=1).to(torch.long)

        # 3) Server KD 训练（CE + alpha * T^2 * KL）
        self.server_model.train()
        T = float(self.cfg.kd_temperature)
        alpha = float(self.cfg.kd_alpha)
        kd_loss_fn = KLDivergenceWithTemperature(temperature=T)

        opt = torch.optim.SGD(
            self.server_model.parameters(), lr=self.cfg.lr,
            momentum=self.cfg.momentum, weight_decay=5e-4
        )

        # 为了对齐顺序，public_loader 必须 shuffle=False
        for ep in range(self.cfg.server_kd_epochs):
            total = 0
            loss_sum = 0.0
            correct = 0
            offset = 0
            for x in tqdm(self.public_loader, desc=f"Server KD {ep+1}/{self.cfg.server_kd_epochs}", leave=False):
                x = x.to(self.device)
                bsz = x.size(0)

                y_p = pseudo[offset: offset + bsz].to(self.device)                 # CE 标签
                t_logits = agg_logits[offset: offset + bsz].to(self.device)         # teacher logits（聚合后）
                offset += bsz

                opt.zero_grad()
                s_logits = self.server_model(x)

                ce = torch.nn.functional.cross_entropy(s_logits, y_p)
                kl = kd_loss_fn(s_logits, t_logits)
                loss = (1 - alpha) * ce + alpha * kl
                loss.backward()
                opt.step()

                loss_sum += loss.item() * bsz
                pred = s_logits.argmax(dim=1)
                correct += (pred == y_p).sum().item()
                total += bsz

            logger.info(
                f"[ServerKD][Round {self.current_round}] Epoch {ep+1}/{self.cfg.server_kd_epochs} "
                f"loss={(loss_sum/total):.4f}, acc={(correct/total):.4f}"
            )

        # 4) 生成 teacher logits（由 server 模型出）
        self.server_model.eval()
        out_logits = []
        with torch.no_grad():
            for x in tqdm(self.public_loader, desc="Server infer teacher logits", leave=False):
                x = x.to(self.device)
                out = self.server_model(x)  # [B,C]
                out_logits.append(out.cpu())
        out_full = torch.cat(out_logits, dim=0).contiguous()  # [N,C]
        self.teacher_logits_bytes = out_full.numpy().astype("float32").tobytes()

        # 5) （可选）评测 server_test_loader
        if self.server_test_loader is not None:
            self._eval_on_server_test()

    def _eval_on_server_test(self):
        self.server_model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in tqdm(self.server_test_loader, desc="Server Eval", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.server_model(x)
                loss = F.cross_entropy(logits, y)
                loss_sum += loss.item() * y.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        logger.info(f"[Round {self.current_round}] Server Eval — loss={loss_sum/total:.4f}, acc={correct/total:.4f}")

    # ---------- 客户端 KD 完成 ACK ----------
    def accept_kd_ack(self, client_id: str, round_id: int) -> bool:
        with self.lock:
            if round_id != self.current_round:
                logger.warning(f"Drop KD ACK from {client_id}: stale round={round_id} (curr={self.current_round})")
                return False
            self.kd_acks.add(client_id)
            logger.info(
                f"[AggregatorKD] recv KD ACK from {client_id} for round={round_id} "
                f"acks={len(self.kd_acks)}/{self.expected_clients}"
            )
            if len(self.kd_acks) >= self.expected_clients:
                # 进入下一轮
                self._advance_round()
            return True

    def _advance_round(self):
        """收齐 ACK 后推进轮次；最后一轮直接进入 DONE。"""
        # 先清理当前轮缓存
        self.received_logits.clear()
        self.kd_acks.clear()
        self.teacher_logits_bytes = None
        self.teacher_sent_to.clear()

        next_round = self.current_round + 1
        if next_round >= self.cfg.total_rounds:
            # 训练完成：进入 DONE，相位锁死
            logger.info(f"[Round {self.current_round}] Completed. All {self.cfg.total_rounds} rounds finished.")
            self.current_round = next_round  # 让 current_round == total_rounds，避免边界歧义
            self.phase = fed_pb2.PHASE_DONE
        else:
            logger.info(f"[Round {self.current_round}] Completed. Advancing to round {next_round}.")
            self.current_round = next_round
            self.phase = fed_pb2.PHASE_LOCAL_SUP