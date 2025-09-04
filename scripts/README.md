python scripts/launch.py \
  --num_clients 20 \
  --bind 0.0.0.0:50052 \
  --server_addr 127.0.0.1:50052 \
  --data_root ./data \
  --dataset_name cifar10 \
  --rounds 30 \
  --local_epochs 5 \
  --batch_size 64 \
  --lr 0.01 \
  --momentum 0.9 \
  --partition_method dirichlet \
  --dirichlet_alpha 0.5 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name resnet18 \
  --max_message_mb 128 \
  --server_warmup_sec 2 \
  --stagger_sec 0.2 \
  --env_omp1


可加 --num_worker 0