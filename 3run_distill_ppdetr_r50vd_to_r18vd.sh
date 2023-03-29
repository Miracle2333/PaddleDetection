export FLAGS_allocator_strategy=auto_growth
job_name=distill_stu_ppdino_r18vd_12e


#"-c", "configs/dino/ppdino_distill/distill_stu_ppdino_r18vd_12e.yml",
#"--slim_config=configs/dino/ppdino_distill/distill_tea_ppdino_r50vd_to_r18vd_12e_kd_teach_detr.yml",
config=configs/dino/ppdino_distill/distill_stu_ppdino_r18vd_12e.yml
#tea_config=configs/dino/ppdino_distill/distill_tea_ppdino_r50vd_to_r18vd_12e.yml
tea_config=configs/dino/ppdino_distill/distill_tea_ppdino_r50vd_to_r18vd_12e_kd_teach_detr.yml

log_dir=log_dir/distill_stu_ppdino_r18vd_12e_kddetr
weights=output/distill_tea_ppdino_r50vd_to_r18vd_12e_kd_teach_norm/0.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp
nohup python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3 tools/train.py -c ${config} --slim_config ${tea_config} --eval >/dev/null 2>&1 &

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp
