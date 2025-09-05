# server/aggregator.py
import logging
import threading
from typing import Dict, List, Tuple, Optional, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from proto import fed_pb2
from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.model.create_model import create_model

logger = logging.getLogger("Server").getChild("Aggregator")


class Aggregator:
    """
    FedNew 强同步状态机：
      PHASE_LOCAL_TRAIN  : 全体 client 拉取上一轮全局小模型 -> 本地训练 -> 上传(带 group_id)
                           收齐 -> 组内聚合 -> PHASE_GROUP_AGG
      PHASE_GROUP_AGG    : 按组下发“该组小模型聚合结果”；转入 PHASE_GROUP_LOGITS
      PHASE_GROUP_LOGITS : 每组仅 1 名代表在公共集上算 logits -> 上传
                           收齐 group_num 份 logits -> Server KD（大模型）
      PHASE_CLIENT_KD    : 下发 server 的 teacher logits 给所有 client，客户端做 KD -> 回 ACK
      -> 进入下一轮（round += 1）
    """
    def __init__(
        self,
        cfg,
        public_unl_loader,
        server_test_loader: Optional[torch.utils.data.DataLoader] = None,
        public_labels: Optional[torch.LongTensor] = None,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)

        # 小模型（客户端用）
        self.client_model = create_model(self.cfg.model_name, num_classes=self.cfg.num_classes).to(self.device)
        self.global_bytes = state_dict_to_bytes(self.client_model.state_dict())

        # 大模型（server KD 用）
        self.server_model = create_model(self.cfg.server_model_name, num_classes=self.cfg.num_classes).to(self.device)

        # 数据
        self.public_unl_loader = public_unl_loader   # 仅图像，要求 shuffle=False
        self.server_test_loader = server_test_loader # 可选评测集 (x,y)

        # —— 公共集真标签：如果外部传入则使用，否则视为未知（训练继续，但跳过伪标签准确率统计）——
        if public_labels is not None:
            self.public_labels = torch.as_tensor(public_labels, dtype=torch.long, device="cpu").contiguous()
            self.public_labels_known = True
            logger.info(f"[Public] labels provided: N={len(self.public_labels)}")
        else:
            self.public_labels = torch.empty(0, dtype=torch.long)
            self.public_labels_known = False
            logger.warning("[Public] labels not provided — pseudo-label accuracies will be skipped.")

        # 状态
        self.current_round = 0
        self.phase: int = fed_pb2.PHASE_LOCAL_TRAIN

        self.registered: List[str] = []
        self.client_index: Dict[str, int] = {}
        self.client_group: Dict[str, int] = {}

        # 本轮提交缓存
        self.received_updates: Dict[str, Tuple[bytes, int, int]] = {}  # cid -> (bytes, num_samples, group_id)
        self.completed_this_round: Set[str] = set()

        # 组内聚合结果（小模型）
        self.group_agg_bytes: Dict[int, bytes] = {}

        # 每组成员 / logits / 代表
        self.group_members: Dict[int, List[str]] = {}    # gid -> [cid...]
        self.group_logits: Dict[int, torch.Tensor] = {}  # gid -> [N,C]
        self.server_logits: Optional[torch.Tensor] = None

        # 本轮每组代表缓存 & 是否已公告
        self.group_rep_for_round: Dict[int, str] = {}
        self._reps_announced_round: int = -1

        # —— 统计：伪标签准确率 —— #
        # round -> { gid: (rep_client_id, acc) }
        self.round_group_pseudo_acc: List[Dict[int, Tuple[str, float]]] = []
        # round -> server 平均 logits 的伪标签准确率
        self.round_server_avglogits_acc: List[float] = []
        # round -> KD 后 server 大模型在公共集上的伪标签准确率
        self.round_server_model_acc: List[float] = []

        # —— 统计：每轮服务器评测（来自 server_test_loader） —— #
        self.server_eval_acc: List[float] = []
        self.server_eval_loss: List[float] = []

        self.lock = threading.Lock()

    # ---------- 工具 ----------
    def ready_to_start(self) -> bool:
        return len(self.registered) >= self.cfg.num_clients

    def is_all_done(self) -> bool:
        with self.lock:
            return self.current_round >= self.cfg.total_rounds and self.phase == fed_pb2.PHASE_DONE

    # ---------- 注册 ----------
    def register(self, client_name: str):
        with self.lock:
            cid = f"C{len(self.registered):03d}"
            idx = len(self.registered)
            self.registered.append(cid)
            self.client_index[cid] = idx
            gid = idx % self.cfg.group_num
            self.client_group[cid] = gid
            self.group_members.setdefault(gid, []).append(cid)
            logger.info(f"Registered {cid} (index={idx}, group={gid})")
            return cid, idx, gid

    # ---------- 下发任务 ----------
    def get_task(self, client_id: str):
        with self.lock:
            if self.current_round >= self.cfg.total_rounds:
                return self.current_round, fed_pb2.PHASE_DONE, False, b"", b""
            if not self.ready_to_start():
                return self.current_round, fed_pb2.PHASE_WAITING, False, b"", b""

            phase = self.phase
            participate = True
            model_bytes = b""
            server_logits_bytes = b""

            if phase == fed_pb2.PHASE_LOCAL_TRAIN:
                model_bytes = self.global_bytes

            elif phase == fed_pb2.PHASE_GROUP_AGG:
                gid = self.client_group[client_id]
                buf = self.group_agg_bytes.get(gid, None)
                if buf is None:
                    participate = False
                else:
                    model_bytes = buf

            elif phase == fed_pb2.PHASE_GROUP_LOGITS:
                gid = self.client_group[client_id]
                rep = self.group_rep_for_round.get(gid)
                participate = (rep == client_id) if rep is not None else False

            elif phase == fed_pb2.PHASE_CLIENT_KD:
                if self.server_logits is not None:
                    server_logits_bytes = self.server_logits.cpu().numpy().astype("float32").tobytes()
                else:
                    participate = False

            return self.current_round, phase, participate, model_bytes, server_logits_bytes

    # ---------- 提交本地小模型 ----------
    def submit_update(self, client_id: str, group_id: int, round_id: int, local_bytes: bytes, num_samples: int):
        with self.lock:
            if round_id != self.current_round or self.phase != fed_pb2.PHASE_LOCAL_TRAIN:
                return False
            if client_id in self.received_updates:
                return True

            self.received_updates[client_id] = (local_bytes, int(num_samples), int(group_id))
            self.completed_this_round.add(client_id)
            logger.info(f"[Round {self.current_round}] recv update {len(self.received_updates)}/{self.cfg.num_clients} from {client_id}")

            if len(self.received_updates) >= self.cfg.num_clients:
                self._group_aggregate_locked()
                self.phase = fed_pb2.PHASE_GROUP_AGG
                logger.info(f"[Round {self.current_round}] -> PHASE_GROUP_AGG (group models ready)")
                self._enter_phase_group_logits_locked()
            return True

    def _enter_phase_group_logits_locked(self):
        self.phase = fed_pb2.PHASE_GROUP_LOGITS
        self.group_logits.clear()

        # 预选代表（round-robin）
        self.group_rep_for_round.clear()
        for gid, members in self.group_members.items():
            if members:
                idx = self.current_round % len(members)
                self.group_rep_for_round[gid] = members[idx]

        # 本轮只公告一次
        if self._reps_announced_round != self.current_round:
            logger.info(f"[Round {self.current_round}] -> PHASE_GROUP_LOGITS (waiting group reps)")
            for gid in sorted(self.group_rep_for_round.keys()):
                rep = self.group_rep_for_round[gid]
                logger.info(f"[Round {self.current_round}] PHASE_GROUP_LOGITS: group={gid} representative={rep}")
            self._reps_announced_round = self.current_round

    def _group_aggregate_locked(self):
        per_group: Dict[int, List[Tuple[bytes, int]]] = {}
        for cid, (b, n, gid) in self.received_updates.items():
            per_group.setdefault(gid, []).append((b, n))

        self.group_agg_bytes.clear()
        for gid, items in per_group.items():
            sds, ws = [], []
            for b, n in items:
                sds.append(bytes_to_state_dict(b))
                ws.append(float(n))
            total = sum(ws) if sum(ws) > 0 else float(len(ws))
            tmpl = sds[0]
            agg = {}
            for k, v in tmpl.items():
                if v.dtype.is_floating_point:
                    acc = torch.zeros_like(v, dtype=torch.float32)
                    for sd, w in zip(sds, ws):
                        acc += sd[k].to(torch.float32) * (w / total)
                    agg[k] = acc.to(v.dtype)
                else:
                    agg[k] = v.clone()
            self.group_agg_bytes[gid] = state_dict_to_bytes(agg)
            logger.info(f"[Round {self.current_round}] Group {gid} aggregated {len(items)} updates.")

        # 刷新下一轮起始小模型（各组均值）
        self._refresh_global_from_groups_locked()

    def _refresh_global_from_groups_locked(self):
        if not self.group_agg_bytes:
            return
        sds = [bytes_to_state_dict(b) for b in self.group_agg_bytes.values()]
        tmpl = sds[0]
        agg = {}
        for k, v in tmpl.items():
            if v.dtype.is_floating_point:
                acc = torch.zeros_like(v, dtype=torch.float32)
                for sd in sds:
                    acc += sd[k].to(torch.float32) / float(len(sds))
                agg[k] = acc.to(v.dtype)
            else:
                agg[k] = v.clone()
        self.client_model.load_state_dict(agg)
        self.global_bytes = state_dict_to_bytes(self.client_model.state_dict())

    # ---------- 接收组代表上传的 logits（并计算该代表的伪标签准确率） ----------
    def accept_group_logits(self, client_id: str, round_id: int, logits_bytes: bytes, num_classes: int, total_examples: int):
        with self.lock:
            if round_id != self.current_round or self.phase != fed_pb2.PHASE_GROUP_LOGITS:
                return
            gid = self.client_group[client_id]

            # 只保存，不做任何“单组提前打伪标签”的计算/打印
            arr = np.frombuffer(logits_bytes, dtype=np.float32).copy()  # 复制规避只读
            if arr.size != total_examples * num_classes:
                logger.error(f"[Round {self.current_round}] bad logits size from {client_id}: {arr.size} vs {total_examples*num_classes}")
                return
            t = torch.from_numpy(arr).view(total_examples, num_classes).contiguous()
            self.group_logits[gid] = t

            logger.info(f"[Round {self.current_round}] received group logits "
                        f"{len(self.group_logits)}/{self.cfg.group_num}")

            # 只有在“收齐所有组”之后，才进行任何伪标签相关的计算与KD
            if len(self.group_logits) >= self.cfg.group_num:
                # 1) 先平均
                avg_logits = torch.stack([self.group_logits[g] for g in sorted(self.group_logits.keys())], dim=0).mean(dim=0)

                # 2) 统一（可选）统计：先统计每组代表伪标签准确率，再统计“平均logits”的伪标签准确率
                if self.public_labels_known:
                    N = min(len(self.public_labels), avg_logits.size(0))
                    # 每组代表（集中统计/打印）
                    per_round_map: Dict[int, Tuple[str, float]] = {}
                    for g in sorted(self.group_logits.keys()):
                        pred_g = self.group_logits[g].argmax(dim=1)[:N]
                        acc_g = (pred_g == self.public_labels[:N]).float().mean().item()
                        # 找到该组代表是谁（用于记录），没缓存就用 "unknown"
                        rep_id = self.group_rep_for_round.get(g, "unknown")
                        per_round_map[g] = (rep_id, acc_g)
                    # 保存整轮的 group 伪标签 acc
                    if len(self.round_group_pseudo_acc) <= self.current_round:
                        self.round_group_pseudo_acc.append(per_round_map)
                    else:
                        self.round_group_pseudo_acc[self.current_round] = per_round_map
                    # 统一打印
                    for g in sorted(per_round_map.keys()):
                        rep_id, acc_g = per_round_map[g]
                        logger.info(f"[Round {self.current_round}] group={g} rep={rep_id} pseudo-acc={acc_g:.4f} ({N} samples)")

                    # 平均 logits 的伪标签 acc
                    pred_avg = avg_logits.argmax(dim=1)[:N]
                    acc_avg = (pred_avg == self.public_labels[:N]).float().mean().item()
                    if len(self.round_server_avglogits_acc) <= self.current_round:
                        self.round_server_avglogits_acc.append(acc_avg)
                    else:
                        self.round_server_avglogits_acc[self.current_round] = acc_avg
                    logger.info(f"[Round {self.current_round}] server-avg-logits pseudo-acc={acc_avg:.4f}")
                else:
                    logger.info(f"[Round {self.current_round}] all group logits received; labels unknown -> skip pseudo-acc")

                # 3) 进入 KD（用“平均 logits”打伪标签做 teacher）
                self._server_kd_locked(avg_logits=avg_logits)
                self.phase = fed_pb2.PHASE_CLIENT_KD
                logger.info(f"[Round {self.current_round}] -> PHASE_CLIENT_KD (broadcast server logits)")
                self.completed_this_round.clear()

    # ---------- Server KD（含 tqdm），并计算 KD 后大模型的伪标签准确率 ----------
    def _server_kd_locked(self, avg_logits: torch.Tensor):
        # 1) 伪标签
        pseudo = avg_logits.argmax(dim=1).to(torch.long)

        # 2) Server KD 训练（CE + alpha*KL）
        T = float(getattr(self.cfg, "kd_temperature", 1.0))
        alpha = float(getattr(self.cfg, "kd_alpha", 0.5))
        epochs = int(getattr(self.cfg, "server_kd_epochs", 3))
        lr = float(self.cfg.lr)
        momentum = float(self.cfg.momentum)

        self.server_model.train()
        opt = torch.optim.SGD(self.server_model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

        for ep in range(epochs):
            total, loss_sum, correct = 0, 0.0, 0
            offset = 0
            pbar = tqdm(self.public_unl_loader, desc=f"Server KD {ep+1}/{epochs}", leave=False)
            for batch in pbar:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                bsz = x.size(0)

                t_logits = avg_logits[offset:offset+bsz].to(self.device)
                y_pseudo = pseudo[offset:offset+bsz].to(self.device)
                offset += bsz

                opt.zero_grad()
                s_logits = self.server_model(x)
                ce = F.cross_entropy(s_logits, y_pseudo)
                kl = F.kl_div(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)
                loss = (1.0 - alpha) * ce + alpha * kl
                loss.backward()
                opt.step()

                loss_sum += loss.item() * bsz
                pred = s_logits.argmax(dim=1)
                correct += (pred == y_pseudo).sum().item()
                total += bsz

                pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")

            logger.info(f"[ServerKD][Round {self.current_round}] Epoch {ep+1}/{epochs} "
                        f"loss={loss_sum/total:.4f}, acc={correct/total:.4f}")

        # 3) KD 后，用大模型重新在公共集推理 teacher logits，下发给客户端
        self.server_model.eval()
        logits_out: List[torch.Tensor] = []
        with torch.no_grad():
            pbar_inf = tqdm(self.public_unl_loader, desc="Server infer teacher logits", leave=False)
            for batch in pbar_inf:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                out = self.server_model(x)
                logits_out.append(out.cpu())
        teacher_full = torch.cat(logits_out, dim=0).contiguous()
        self.server_logits = teacher_full

        # —— KD 后大模型伪标签准确率 —— #
        if self.public_labels_known:
            N = min(len(self.public_labels), teacher_full.size(0))
            pred_model = teacher_full.argmax(dim=1)[:N]
            acc_model = (pred_model == self.public_labels[:N]).float().mean().item()
            if len(self.round_server_model_acc) <= self.current_round:
                self.round_server_model_acc.append(acc_model)
            else:
                self.round_server_model_acc[self.current_round] = acc_model
            logger.info(f"[Round {self.current_round}] server-model pseudo-acc={acc_model:.4f}")
        else:
            logger.info(f"[Round {self.current_round}] server-model logits computed; labels unknown -> skip pseudo-acc")

        # 若有带标签的 server_test_loader，可继续做分类评测，并记录到列表
        if self.server_test_loader is not None:
            loss, acc = self._evaluate_server_model_with_tqdm()
            self.server_eval_loss.append(loss)
            self.server_eval_acc.append(acc)
            logger.info(f"[Round {self.current_round}] Server Eval — loss={loss:.4f}, acc={acc:.4f}")

    def _evaluate_server_model_with_tqdm(self) -> Tuple[float, float]:
        total = 0
        correct = 0
        loss_sum = 0.0
        self.server_model.eval()
        with torch.no_grad():
            pbar = tqdm(self.server_test_loader, desc="Server Eval", leave=False)
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    raise RuntimeError("server_test_loader must provide (x, y) batches.")
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.server_model(x)
                loss = F.cross_entropy(logits, y)
                loss_sum += loss.item() * y.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")
        return (loss_sum / total, correct / total)

    # ---------- 客户端 KD 完成后用 UploadUpdate 作为 ACK（空权重） ----------
    def submit_client_kd_ack(self, client_id: str, round_id: int):
        with self.lock:
            if round_id != self.current_round or self.phase != fed_pb2.PHASE_CLIENT_KD:
                return False
            if client_id in self.completed_this_round:
                return True
            self.completed_this_round.add(client_id)
            logger.info(f"[Round {self.current_round}] client KD ACK from {client_id} "
                        f"{len(self.completed_this_round)}/{self.cfg.num_clients}")
            if len(self.completed_this_round) >= self.cfg.num_clients:
                self._advance_round_locked()
            return True

    def _advance_round_locked(self):
        logger.info(f"[Round {self.current_round}] Completed. -> Round {self.current_round+1}")
        self.current_round += 1
        if self.current_round >= self.cfg.total_rounds:
            self.phase = fed_pb2.PHASE_DONE
        else:
            self.phase = fed_pb2.PHASE_LOCAL_TRAIN
        # 清空（历史统计保留在 round_* / server_eval_* 列表中）
        self.received_updates.clear()
        self.group_agg_bytes.clear()
        self.group_logits.clear()
        self.server_logits = None
        self.completed_this_round.clear()

    def will_accept_update(self, client_id: str, round_id: int) -> bool:
        with self.lock:
            return (
                round_id == self.current_round
                and self.phase == fed_pb2.PHASE_LOCAL_TRAIN
                and client_id not in self.received_updates
            )