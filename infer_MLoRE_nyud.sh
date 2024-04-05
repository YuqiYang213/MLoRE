RANDOM=$$
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port=$((RANDOM%1000+12000))  main.py --config_exp './configs/nyud/nyud_vitLp16_MLoRE_repara.yml' --run_mode infer \
    --trained_model pretrain_nyud_vitl.pth
