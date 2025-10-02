python scripts/launch.py \
  --num_clients 20 \
  --bind 0.0.0.0:51052 \
  --server_addr 127.0.0.1:51052 \
  --data_root ./dataset \
  --dataset_name NWPU-RESISC45 \
  --num_classes 15 \
  --rounds 50 \
  --local_epochs 3 \
  --batch_size 32 \
  --lr 0.005 \
  --momentum 0.9 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name resnet18 \
  --max_message_mb 128 \
  --server_warmup_sec 25 \
  --stagger_sec 0.2 \
  --env_omp1 \
  --gpus 0,1,2,3 \
  --log_dir logs



