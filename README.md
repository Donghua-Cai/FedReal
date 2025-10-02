# FedReal

依赖：pip install torch torchvision torchaudio grpcio grpcio-tools

## 生成 gRPC 代码

在项目根目录（包含 proto/ 文件夹）执行：

python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/fed.proto

将在 proto/ 下生成 fed_pb2.py 与 fed_pb2_grpc.py

注意重新生成后要修改fed_pb2_grpc.py 

把 import fed_pb2 as fed__pb2 改成 from . import fed_pb2 as fed__pb2

## 启动示例

在根目录下运行launch.py，具体命令参照scripts下的README.md

### 实机启动命令

server：

python -m server.server_main \
  --bind 0.0.0.0:50052 \
  --data_root ./dataset \
  --dataset_name Cifar10 \
  --num_classes 10 \
  --num_clients 3 \
  --rounds 50 \
  --local_epochs 3 \
  --batch_size 32 \
  --lr 0.005 \
  --momentum 0.9 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name resnet18 \
  --max_message_mb 128 \
  --device cpu

client:

python -m client.client_main \
  --server 192.168.202.197:50052 \
  --client_name c0 \
  --data_root ./dataset \
  --dataset_name Cifar10 \
  --num_classes 10 \
  --num_clients 3 \
  --batch_size 32 \
  --seed 42