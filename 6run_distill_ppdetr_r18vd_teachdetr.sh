#export FLAGS_allocator_strategy=auto_growth                                                                                                            
job_name=distill_stu_ppdino_r18vd_12e                                                                                                                  
                                                                                                                                                       
config=configs/dino/ppdino_distill/dino_r18vd_pan_3_0_6_12e_coco_teach-detr.yml                                                                                         
#tea_config=configs/dino/ppdino_distill/distill_tea_ppdino_r50vd_to_r18vd_12e.yml                                                                      
#tea_config=configs/dino/ppdino_distill/                                     
                                                                                                                                                       
log_dir=log_dir/distill_stu_ppdino_r18vd_12e                                                                                                           
weights=output/distill_tea_ppdino_r18vd_to_r18vd_12e/0.pdparams                                                                                        
                                                                                                                                                       
# 1. training                                                                                                                                          
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp                                                
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 4,5,6,7 tools/train.py -c ${config} --eval #--amp       
                                                                                                    