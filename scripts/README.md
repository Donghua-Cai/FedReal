python scripts/launch.py \
  --num_clients 20 \
  --bind 0.0.0.0:51052 \
  --server_addr 127.0.0.1:51052 \
  --data_root ./dataset \
  --dataset_name DOTA \
  --num_classes 15 \
  --rounds 10 \
  --local_epochs 5 \
  --batch_size 10 \
  --lr 0.005 \
  --momentum 0.9 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name resnet18 \
  --max_message_mb 128 \
  --server_warmup_sec 10 \
  --stagger_sec 0.2 \
  --env_omp1 \
  --gpus 1 \
  --log_dir logs



