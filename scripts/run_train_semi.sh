export PYTHONPATH=.
nohup python -m paddle.distributed.launch --gpus 1,2,3,4 tools/train.py \
 -c configs/semi_det/semi_detr/dino_ssod_debug.yml \
  --eval --use_vdl=true --vdl_log_dir=vdl_dir/5_lr_10 1>logs/5_sup.log 2>logs/5_sup.err &
