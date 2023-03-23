export PYTHONPATH=.
nohup python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
 -c configs/dino/dino_r50vd_pan_4_0_6_3x_coco.yml \
  --eval --use_vdl=true --vdl_log_dir=vdl_dir/10_full 1>logs/10_sup.log 2>logs/10_sup.err &
