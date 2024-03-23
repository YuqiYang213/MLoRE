RANDOM=$$
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2  --master_port=$((RANDOM%1000+12000))  main.py --config_exp './configs/pascal/pascal_vitLp16_MLoRE.yml' --run_mode train
