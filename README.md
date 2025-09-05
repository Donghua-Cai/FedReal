# FedReal
Real Federated Learning via python grpc

两层（Server–Client）联邦学习最小可用实现：
- 数据集与划分在 `common/dataset/`，支持 **IID** 与 **Dirichlet**；先使用官方 CIFAR-10 *test* 作为公共评测集；每个客户端再从自己的份额中切出本地 *test*（默认 10%）。
- 模型在 `common/model/`，目前仅 `resnet18`（来自 `torchvision`）。
- 传输走 **完整权重**（`state_dict` → bytes），并在 gRPC 里把最大消息大小调大到默认 **128MB**。
- 服务端与客户端都做评测；服务端在每轮聚合后用公共测试集评估。
- 代码层面高度解耦，后续可加三层拓展（边缘聚合、伪标签/蒸馏）。
## 依赖
```bash
pip install torch torchvision torchaudio grpcio grpcio-tools

生成 gRPC 代码

注意重新生成后要修改fed_pb2_grpc.py: from . import fed_pb2 as fed__pb2

# 在项目根目录（包含 proto/ 文件夹）执行：
python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/fed.proto
# 将在 proto/ 下生成 fed_pb2.py 与 fed_pb2_grpc.py

启动示例（单机多进程测试）

1) 启动服务端

python -m server.server_main \
  --bind 0.0.0.0:50052 \
  --data_root ./data \
  --num_clients 2 \
  --rounds 3 \
  --local_epochs 5 \
  --batch_size 64 \
  --lr 0.01 --momentum 0.9 \
  --partition_method iid \
  --dirichlet_alpha 0.5 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name resnet18 \
  --max_message_mb 128

2) 启动 3 个客户端（不同终端/进程）

python -m client.client_main --server 127.0.0.1:50052 --client_name c0 --data_root ./data
python -m client.client_main --server 127.0.0.1:50052 --client_name c1 --data_root ./data
python -m client.client_main --server 127.0.0.1:50052 --client_name c2 --data_root ./data

python -m client.client_main \
  --server 127.0.0.1:50052 \
  --client_name c0 \
  --data_root ./data \
  --dataset_name cifar10 \
  --num_clients 2 \
  --partition_method iid \
  --dirichlet_alpha 0.5 \
  --client_test_ratio 0.1 \
  --batch_size 64

python -m server.server_main   --bind 0.0.0.0:50052   --data_root ./data   --dataset_name cifar10   --num_clients 2   --rounds 3   --local_epochs 2   --batch_size 64   --lr 0.01   --momentum 0.9   --partition_method iid   --dirichlet_alpha 0.5   --sample_fraction 1.0   --seed 42   --model_name resnet18   --max_message_mb 128

python -m client.client_main   --server 127.0.0.1:50052   --client_name c0   --data_root ./data   --dataset_name cifar10   --num_clients 2   --partition_method iid   --dirichlet_alpha 0.5   --client_test_ratio 0.1   --batch_size 64

python -m client.client_main   --server 127.0.0.1:50052   --client_name c1   --data_root ./data   --dataset_name cifar10   --num_clients 2   --partition_method iid   --dirichlet_alpha 0.5   --client_test_ratio 0.1   --batch_size 64


Dirichlet

python -m server.server_main   --bind 0.0.0.0:50052   --data_root ./data   --dataset_name cifar10   --num_clients 2   --rounds 3   --local_epochs 5   --batch_size 64   --lr 0.01   --momentum 0.9   --partition_method dirichlet   --dirichlet_alpha 0.5   --sample_fraction 1.0   --seed 42   --model_name resnet18   --max_message_mb 128

python -m client.client_main   --server 127.0.0.1:50052   --client_name c0   --data_root ./data   --dataset_name cifar10   --num_clients 2   --partition_method dirichlet   --dirichlet_alpha 0.5   --client_test_ratio 0.1   --batch_size 64

python -m client.client_main   --server 127.0.0.1:50052   --client_name c1   --data_root ./data   --dataset_name cifar10   --num_clients 2   --partition_method dirichlet   --dirichlet_alpha 0.5   --client_test_ratio 0.1   --batch_size 64


Label distribution: {7: 3582, 3: 1326, 4: 1698, 2: 1960, 5: 445, 8: 806, 1: 1491, 6: 1658, 0: 511, 9: 90}

Label distribution: {2: 2555, 0: 4024, 3: 3174, 5: 4033, 9: 4398, 1: 2988, 6: 2830, 8: 3705, 4: 2787, 7: 939}


2025-09-04 16:53:43 [INFO] Server: Server total acc : [0.351, 0.736, 0.752]
2025-09-04 16:53:43 [INFO] Server: Server total loss : [1.7694936513900756, 0.7867589998245239, 0.738979609489441]


2025-09-04 16:53:42 [INFO] Client-c0: [Client C000] total acc : [0.5527934392619169, 0.6158380317785751, 0.7091235263967196]
2025-09-04 16:53:42 [INFO] Client-c0: [Client C000] total loss : [1.2718044392944055, 1.1273983765235014, 0.8592933611586301]


2025-09-04 16:53:43 [INFO] Client-c1: [Client C001] total acc : [0.6412318556296587, 0.6834052569635151, 0.7326402510788544]
2025-09-04 16:53:43 [INFO] Client-c1: [Client C001] total loss : [1.1695440298859023, 0.9748998435397577, 0.7859127272480186]