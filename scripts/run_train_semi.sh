export PYTHONPATH=.
nohup python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py \
 -c configs/semi_det/semi_detr/dino_ssod_gthr.yml \
  --eval --use_vdl=true --vdl_log_dir=vdl_dir/thres 1>logs/thres.log 2>logs/thres.err &
