python scripts/launch.py \
  --num_clients 20 \
  --bind 0.0.0.0:50052 \
  --server_addr 127.0.0.1:50052 \
  --data_root ./dataset \
  --dataset_name Cifar10 \
  --num_classes 10 \
  --rounds 50 \
  --local_epochs 5 \
  --batch_size 64 \
  --lr 0.01 \
  --momentum 0.9 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name resnet18 \
  --max_message_mb 128 \
  --server_warmup_sec 2 \
  --stagger_sec 0.2 \
  --env_omp1 \
  --gpus 4,5,6,7 \
  --log_dir logs



