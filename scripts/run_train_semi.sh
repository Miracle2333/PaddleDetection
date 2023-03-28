export PYTHONPATH=.
nohup python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 --log_dir log_dir/semi_sslad_10 tools/train.py \
 -c configs/semi_det/semi_detr/dino_ssod_sslad.yml \
  --eval --use_vdl=true --vdl_log_dir=vdl_dir/ssod_sslad >/dev/null 2>&1 &
