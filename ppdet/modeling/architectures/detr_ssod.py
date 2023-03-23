# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
# 
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import KeysView
import copy
from ppdet.core.workspace import register, create, merge_config
from .meta_arch import BaseArch
from ppdet.data.reader import transform
import paddle
import os

import numpy as np
from operator import itemgetter
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.fluid import framework
from ppdet.core.workspace import register, create
# from ppdet.data.reader import get_dist_info
# from ppdet.modeling.proposal_generator.target import label_box
from ppdet.modeling.bbox_utils import delta2bbox
from ppdet.data.transform.atss_assigner import bbox_overlaps
from ppdet.utils.logger import setup_logger
from ppdet.modeling.ssod.utils import filter_invalid, weighted_loss
from .multi_stream_detector import MultiSteamDetector
logger = setup_logger(__name__)

__all__ = ['DETR_SSOD']
@register
class DETR_SSOD(MultiSteamDetector):
    def __init__(self, teacher, student, train_cfg=None, test_cfg=None, PPDETRTransformer=None):
        super(DETR_SSOD, self).__init__(
            dict(teacher=teacher, student=student),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.semi_start_iters=train_cfg['semi_start_iters']
        self.ema_start_iters=train_cfg['ema_start_iters']
        self.momentum=0.9996
        self.cls_thr=None
        self.cls_thr_ig=None
        # self.id=0
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg['unsup_weight']
            self.sup_weight = self.train_cfg['sup_weight']
            self._teacher = None
            self._student = None
            self._transformer = None

    @classmethod
    def from_config(cls, cfg):
        teacher = create(cfg['teacher'])
        merge_config(cfg)
        student = create(cfg['student'])
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        PPDETRTransformer = cfg['PPDETRTransformer']
        return {
            'teacher': teacher,
            'student': student,
            'train_cfg': train_cfg,
            'test_cfg' : test_cfg,
            'PPDETRTransformer': PPDETRTransformer
        }

    def forward_train(self, inputs, **kwargs):
        if isinstance(inputs,dict):
            iter_id=inputs['iter_id']
        elif isinstance(inputs,list):
            iter_id=inputs[-1]
        if iter_id==self.semi_start_iters:
            self.update_ema_model(momentum=0)
        elif iter_id>self.semi_start_iters:
            self.update_ema_model(momentum=self.momentum)
        # elif iter_id<self.semi_start_iters:
        #     self.update_ema_model(momentum=0)
        if iter_id>=self.semi_start_iters:
            if iter_id==self.semi_start_iters:
                print('***********************')
                print('******semi start*******')
                print('***********************')
            data_sup_w, data_sup_s, data_unsup_w, data_unsup_s,_=inputs
            
            if data_sup_w['image'].shape != data_sup_s['image'].shape:
                data_sup_w, data_sup_s = align_weak_strong_shape(data_sup_w,data_sup_s)
                                                                                    
            if  'gt_bbox' in data_unsup_s.keys():
                del data_unsup_s['gt_bbox']
            if  'gt_class' in data_unsup_s.keys():
                del data_unsup_s['gt_class']  
            if  'gt_class' in data_unsup_w.keys():
                del data_unsup_w['gt_class']  
            if  'gt_bbox' in data_unsup_w.keys():
                del data_unsup_w['gt_bbox']    
            for k, v in data_sup_s.items():
                if k in ['epoch_id']:
                    continue
                elif k in ['gt_class','gt_bbox','is_crowd']:
                    data_sup_s[k].extend(data_sup_w[k])
                else:
                    data_sup_s[k] = paddle.concat([v,data_sup_w[k]])
            # print('***********unsup_w**************')
            # print(data_unsup_w)
            # print('***********unsup_s**************')
            # print(data_unsup_s)
            loss={}
            body_feats=self.student.backbone(data_sup_s)
            if self.student.neck is not None:
                    body_feats = self.student.neck(body_feats)
            out_transformer = self.student.transformer(body_feats,data_sup_s)
            sup_loss = self.student.detr_head(out_transformer, body_feats, data_sup_s)
            sup_loss.update({
                'loss':
                paddle.add_n([v for k, v in sup_loss.items() if 'log' not in k])
            })
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}

            loss.update(**sup_loss)   
            unsup_loss =  self.foward_unsup_train(data_unsup_w, data_unsup_s)
            unsup_loss.update({
            'loss':
            paddle.add_n([v for k, v in unsup_loss.items() if 'log' not in k])
        })
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            unsup_loss.update({
                'loss':
                paddle.add_n([v for k, v in unsup_loss.items() if 'log' not in k])
            })
            loss.update(**unsup_loss)      
            loss.update({'loss': loss['sup_loss'] + loss['unsup_loss'] })
        else:
            if iter_id==self.semi_start_iters-1:
                print('********************')
                print('******sup ing*******')
                print('********************')
            loss = {}
            sup_loss=self.student(inputs)
            sup_loss = {k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)      

        return loss

    def foward_unsup_train(self, data_unsup_w, data_unsup_s):

        with paddle.no_grad():
            body_feats=self.teacher.backbone(data_unsup_w)
            if self.teacher.neck is not None:
                body_feats = self.teacher.neck(body_feats,is_teacher=True)
            out_transformer = self.teacher.transformer(body_feats, data_unsup_w,is_teacher=True)
            preds = self.teacher.detr_head(out_transformer, body_feats,is_teacher=True)
            bbox=preds[0].astype('float32')
            label=preds[1].argmax(-1).unsqueeze(-1).astype('float32')
            score=F.softmax(preds[1],axis=2).max(-1).unsqueeze(-1).astype('float32')
            iou_score=preds[-1]
            bs,bbox_num=bbox.shape[:2]
            bbox_num=paddle.tile(paddle.to_tensor([bbox_num]), [bs])
        self.place=body_feats[0].place


        bboxes=paddle.concat([label,score,bbox,iou_score],axis=-1).reshape([-1,7])
        # print(score.max())    

        proposal_score_list = bboxes[:,1:2].squeeze(-1)
        proposal_score_list  = proposal_score_list.split(tuple(np.array(bbox_num)), 0)
        
        proposal_label_list = paddle.cast(bboxes[:, 0], np.int32)
        proposal_label_list = proposal_label_list.split(tuple(np.array(bbox_num)), 0)

        proposal_bbox_list = bboxes[:,2:6]
        proposal_bbox_list  = proposal_bbox_list.split(tuple(np.array(bbox_num)), 0)
        
        proposal_iou_list = paddle.cast(bboxes[:, -1], np.float32)
        proposal_iou_list = proposal_iou_list.split(tuple(np.array(bbox_num)), 0)     
            
        proposal_score_list = [paddle.to_tensor(p, place=self.place) for p in proposal_score_list]
        proposal_label_list = [paddle.to_tensor(p, place=self.place) for p in proposal_label_list]
        proposal_bbox_list = [paddle.to_tensor(p, place=self.place) for p in proposal_bbox_list]
        proposal_iou_list = [paddle.to_tensor(p, place=self.place) for p in proposal_iou_list]
        
        if isinstance(self.train_cfg['pseudo_label_initial_score_thr'], float):
            thr = self.train_cfg['pseudo_label_initial_score_thr']
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.") 
        proposal_bbox_list, proposal_label_list, proposal_score_list,proposal_iou_list = list(
            zip(
                *[
                    filter_invalid(
                        proposal_bbox,
                        proposal_label,
                        proposal_iou,
                        proposal_score,
                        thr=thr,
                        min_size=self.train_cfg['min_pseduo_box_size'],
                    )
                    for proposal_bbox, proposal_label,proposal_score,proposal_iou in zip(
                        proposal_bbox_list, proposal_label_list, proposal_score_list,proposal_iou_list
                    )
                ]
            )
        )

        teacher_bboxes = list(proposal_bbox_list)
        teacher_labels = proposal_label_list
        teacher_scores = proposal_score_list
        teacher_ious = proposal_iou_list
        teacher_info=[teacher_bboxes,teacher_labels,teacher_scores,teacher_ious]
        student_unsup=data_unsup_s
        return self.compute_pseudo_label_loss(student_unsup, teacher_info)

    def compute_pseudo_label_loss(self,student_unsup, teacher_info):                                 

        pseudo_bboxes=list(teacher_info[0])
        pseudo_labels=list(teacher_info[1])
        pseudo_scores=list(teacher_info[2])
        pseudo_ious=list(teacher_info[3])
        losses = dict()
        for i in range(len(pseudo_bboxes)):
            if pseudo_labels[i].shape[0]==0:
                pseudo_bboxes[i]=paddle.zeros([0,4]).numpy()
                pseudo_labels[i]=paddle.zeros([0,1]).numpy()
                pseudo_scores[i]=paddle.zeros([0,1]).numpy()
                pseudo_ious[i]=paddle.zeros([0,1]).numpy()
            else:
                pseudo_bboxes[i]=pseudo_bboxes[i][:,:4].numpy()
                pseudo_labels[i]=pseudo_labels[i].unsqueeze(-1).numpy()
                pseudo_scores[i]=pseudo_scores[i].unsqueeze(-1).numpy()
                pseudo_ious[i]=pseudo_ious[i].unsqueeze(-1).numpy()
        for i in range(len(pseudo_bboxes)):
            pseudo_labels[i]= paddle.to_tensor(pseudo_labels[i],dtype=paddle.int32,place=self.place)
            pseudo_bboxes[i]= paddle.to_tensor(pseudo_bboxes[i],dtype=paddle.float32,place=self.place)
            pseudo_scores[i]= paddle.to_tensor(pseudo_scores[i],dtype=paddle.float32,place=self.place)
            pseudo_ious[i]= paddle.to_tensor(pseudo_ious[i],dtype=paddle.float32,place=self.place)
        # print(pseudo_bboxes[0].shape[0])
        student_unsup.update({'gt_bbox':pseudo_bboxes,'gt_class':pseudo_labels,'gt_iou':pseudo_ious,'gt_score':pseudo_scores})
        # student_data.update(gt_bbox=pseudo_bboxes,gt_class=pseudo_labels)
        pseudo_sum=0
        # self.id+=1
        for i in range(len(pseudo_bboxes)):
            pseudo_sum+=pseudo_bboxes[i].sum()
        # print(self.id)
        if pseudo_sum==0:
            # print('pseudo_sum=0')
            pseudo_bboxes[0]=paddle.ones([1,4])-0.5
            pseudo_labels[0]=paddle.ones([1,1]).astype('int32')
            student_unsup.update({'gt_bbox':pseudo_bboxes,'gt_class':pseudo_labels})
            body_feats=self.student.backbone(student_unsup)
            if self.student.neck is not None:
                    body_feats = self.student.neck(body_feats)
            out_transformer = self.student.transformer(body_feats,student_unsup)
            losses = self.student.detr_head(out_transformer, body_feats, student_unsup)
  
            for k,v in losses.items():
                losses[k]=v*0.0
            print(losses)
            
        else:
            gt_bbox=[]
            gt_class=[]
            gt_score=[]
            gt_iou=[]
            images=[]
            for i in range(len(pseudo_bboxes)):
                if pseudo_labels[i].shape[0]==0:
                    continue
                else:
                    gt_class.append(pseudo_labels[i])
                    gt_bbox.append(pseudo_bboxes[i])
                    gt_score.append(pseudo_scores[i])
                    gt_iou.append(pseudo_ious[i])                   
                    images.append(student_unsup['image'][i])
            images=paddle.stack(images)
            student_unsup.update({'image':images,'gt_bbox':gt_bbox,'gt_class':gt_class,'gt_iou':gt_iou,'gt_score':gt_score})
            body_feats=self.student.backbone(student_unsup)
            if self.student.neck is not None:
                    body_feats = self.student.neck(body_feats)
            out_transformer = self.student.transformer(body_feats,student_unsup)
            losses = self.student.detr_head(out_transformer, body_feats, student_unsup,is_student=True)
        return losses




def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return paddle.stack(b, axis=-1)

def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (ow, oh)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def align_weak_strong_shape(data_weak, data_strong):
    shape_x = data_strong['image'].shape[2]
    shape_y = data_strong['image'].shape[3]
    
    target_size = [shape_x, shape_y]

    # if scale_x_w != 1 or scale_y_w != 1:
    data_weak['image'] = F.interpolate(
        data_weak['image'],
        size=target_size,
        mode='bilinear',
        align_corners=False)
    return data_weak, data_strong
