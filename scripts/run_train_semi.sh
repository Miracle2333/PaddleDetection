export PYTHONPATH=.
nohup python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py \
 -c configs/semi_det/semi_detr/detr_ssod_lr_10.yml \
  --eval --use_vdl=true --vdl_log_dir=vdl_dir/5_lr_10 1>logs/5_lr_10.log 2>logs/5_lr_10.err &
