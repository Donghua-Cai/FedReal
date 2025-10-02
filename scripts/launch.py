#!/usr/bin/env python3
# scripts/launch_fedreal.py
import argparse
import os
import sys
import time
import shutil
import signal
import subprocess
from datetime import datetime
from pathlib import Path

# —— 预清理：杀掉已有的 server / client 进程 ——
def kill_existing():
    patterns = [
        "python -m server.server_main",
        "python -m client.client_main",
    ]
    for pat in patterns:
        try:
            subprocess.run(["pkill", "-f", pat], check=False)
        except Exception:
            pass

def has_stdbuf() -> bool:
    return shutil.which("stdbuf") is not None

def line_buffer_prefix():
    # 在 Linux/macOS 上优先用 stdbuf 让日志实时写文件
    return ["stdbuf", "-oL", "-eL"] if has_stdbuf() else []

def make_logs_dir():
    Path("logs").mkdir(parents=True, exist_ok=True)

def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def launch(cmd, log_path: Path, env=None):
    log_f = open(log_path, "w")
    # 让子进程成为新的进程组，便于整体 kill（Linux/macOS）
    if os.name != "nt":
        return subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            close_fds=True,
            env=env
        )
    else:
        return subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
            env=env
        )

def kill_proc_tree(proc):
    try:
        if proc.poll() is None:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
    except Exception:
        pass

def build_server_cmd(args):
    py = sys.executable
    cmd = line_buffer_prefix() + [
        py, "-m", "server.server_main",
        "--bind", args.bind,
        "--data_root", args.data_root,
        "--dataset_name", args.dataset_name,
        "--num_classes", str(args.num_classes),
        "--num_clients", str(args.num_clients),
        "--rounds", str(args.rounds),
        "--local_epochs", str(args.local_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--momentum", str(args.momentum),
        "--sample_fraction", str(args.sample_fraction),
        "--seed", str(args.seed),
        "--model_name", args.model_name,
        "--max_message_mb", str(args.max_message_mb),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.num_workers is not None:
        cmd += ["--num_workers", str(args.num_workers)]
    return cmd

def build_client_cmd(args, idx: int):
    py = sys.executable
    client_name = f"c{idx}"
    cmd = line_buffer_prefix() + [
        py, "-m", "client.client_main",
        "--server", args.server_addr,
        "--client_name", client_name,
        "--data_root", args.data_root,
        "--dataset_name", args.dataset_name,
        "--num_classes", str(args.num_classes),
        "--num_clients", str(args.num_clients),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    if args.client_device:
        cmd += ["--device", args.client_device]
    if args.num_workers is not None:
        cmd += ["--num_workers", str(args.num_workers)]
    return cmd, client_name

def parse_args():
    p = argparse.ArgumentParser(description="Launch FedReal server and multiple clients")
    # Topology
    p.add_argument("--num_clients", type=int, default=20)
    # Server config
    p.add_argument("--bind", type=str, default="0.0.0.0:50052")
    p.add_argument("--server_addr", type=str, default="127.0.0.1:50052")
    p.add_argument("--data_root", type=str, default="./dataset")
    p.add_argument("--dataset_name", type=str, default="Cifar10",
                   choices=["Cifar10", "Cifar100", "NWPU-RESISC45", "DOTA"])
    p.add_argument("--num_classes", type=int)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--sample_fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="resnet18")
    p.add_argument("--max_message_mb", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu", help="Server device override, e.g., cuda or cpu")
    p.add_argument("--client_device", type=str, default=None, help="Client device override, e.g., cpu to save GPU")
    p.add_argument("--num_workers", type=int, default=None, help="DataLoader workers for both server/client (if applicable)")

    # Multi-GPU distribution
    p.add_argument("--gpus", type=str, default="0,1,2,3",
                   help="GPU id list for clients, e.g. '0,1,2,3'")
    p.add_argument("--server_gpu", type=int, default=0,
                   help="GPU id for server process (from the real system ids)")

    # Launch behavior
    p.add_argument("--stagger_sec", type=float, default=0.2, help="Stagger between client launches")
    p.add_argument("--server_warmup_sec", type=float, default=2.0, help="Wait before launching clients")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--env_omp1", action="store_true", help="Set OMP/MKL/OPENBLAS threads to 1 to reduce CPU contention")

    return p.parse_args()

def main():
    kill_existing()
    args = parse_args()
    make_logs_dir()

    if args.env_omp1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # 解析 GPU 列表
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if len(gpu_list) == 0:
        print("[Launcher] WARNING: --gpus is empty; clients will inherit parent CUDA_VISIBLE_DEVICES.")
    else:
        print(f"[Launcher] Client GPUs: {gpu_list}")
    print(f"[Launcher] Server GPU: {args.server_gpu}")

    # Server env（放在单独的 GPU 上）
    server_env = os.environ.copy()
    # 若你希望 server 也在 GPU 上跑评测/聚合，可设置为指定 id；否则留空跑 CPU
    if args.device is None or args.device.startswith("cuda"):
        server_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        server_env["CUDA_VISIBLE_DEVICES"] = str(args.server_gpu)
    # 限制碎片化以减缓 OOM 风险
    server_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    # Server
    server_cmd = build_server_cmd(args)
    server_log = Path(args.log_dir) / f"server_{ts()}.log"
    print("[Launcher] Starting Server:")
    print(" ", " ".join(server_cmd))
    print(" ", f"logs -> {server_log}")
    server_proc = launch(server_cmd, server_log, env=server_env)

    # Optional warmup for server to bind port
    time.sleep(args.server_warmup_sec)

    # Clients
    client_procs = []
    for i in range(args.num_clients):
        client_cmd, cname = build_client_cmd(args, i)
        clog = Path(args.log_dir) / f"client_{cname}.log"

        # 为每个 client 设置独立 env
        cenv = os.environ.copy()
        cenv.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        cenv["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        if args.client_device and args.client_device.lower() == "cpu":
            # 强制用 CPU
            cenv["CUDA_VISIBLE_DEVICES"] = ""
            gpu_assigned = "CPU"
        else:
            # 轮转分配 GPU
            if len(gpu_list) > 0:
                gid = gpu_list[i % len(gpu_list)]
                cenv["CUDA_VISIBLE_DEVICES"] = gid
                gpu_assigned = gid
            else:
                # 不改动父进程的 CUDA_VISIBLE_DEVICES
                gpu_assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "<inherit>")

        print(f"[Launcher] Starting Client {cname} on GPU {gpu_assigned}:")
        print(" ", " ".join(client_cmd))
        print(" ", f"logs -> {clog}")
        p = launch(client_cmd, clog, env=cenv)
        client_procs.append(p)
        time.sleep(args.stagger_sec)

    print("\n[Launcher] All processes started.")
    print(f"[Launcher] Tail server log: tail -f {server_log}\n")

    try:
        while True:
            time.sleep(1)
            if server_proc.poll() is not None:
                print("[Launcher] Server exited. Terminating clients...")
                break
    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt received. Stopping all processes...")

    # Cleanup
    for p in client_procs:
        kill_proc_tree(p)
    kill_proc_tree(server_proc)
    print("[Launcher] Done.")

if __name__ == "__main__":
    main()