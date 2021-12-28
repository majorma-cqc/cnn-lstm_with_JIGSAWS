import torch
from torch._C import clear_autocast_cache
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import json
from mean import get_mean, get_std
from PIL import Image
import cv2
from datasets.ucf101 import load_annotation_data
from datasets.ucf101 import get_class_labels
from model import generate_model
from utils import AverageMeter
from opts_JIGSAWS import parse_opts
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose

import matplotlib.pyplot as plt
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def resume_model(opt, model):
    """ Resume model 
    """
    checkpoint = torch.load(opt.resume_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])


def predict(clip, model):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale((150, 150)),
        #Scale(int(opt.sample_size / opt.scale_in_test)),
        #CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    if spatial_transform is not None:
        # spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]
    

    clip = torch.stack(clip, dim=0)
    clip = clip.to(device)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        print(clip.shape)
        #clip.requires_grad = True
        outputs_from_model = model(clip)
        outputs_after_softmax = F.softmax(outputs_from_model)
    
    ''' ----------------- GRAD-CAM Ver 2 -----------------------------------'''
    model.train()
    clip.requires_grad = True
    outputs_from_model_cam = model.activate_cam(clip)
    outputs_after_softmax_cam = F.softmax(outputs_from_model_cam)
    outputs_from_model_cam[:, torch.argmax(outputs_after_softmax_cam)].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(clip).detach()
    for i in range(2048):
        activations[:, i] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = heatmap.cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    #plt.matshow(heatmap.squeeze())
    pass
    
    # heatmap = np.array(heatmap)
    # img = cv2.imread(test_img_path)
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = heatmap * 0.4 + img
    # cv2.imwrite('./grad_dontKnow.jpg', superimposed_img)
    # pass
    
    ''' ----------------- GRAD-CAM Ver 2 -----------------------------------'''

    print(outputs_after_softmax)
    scores, idx = torch.topk(outputs_after_softmax, k=1)
    mask = scores > 0.1
    preds = idx[mask]
    return preds, heatmap


if __name__ == "__main__":
    '''-------------------- TODO: Use pdb debugging --------------------'''
    #import pdb;pdb.set_trace()
    opt = parse_opts()
    print(opt)
    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    device = torch.device("cuda")
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    model = generate_model(opt, device)

    # model = nn.DataParallel(model, device_ids=None)
    # print(model)
    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        model.eval()

        #cam = cv2.VideoCapture(
        #    '/Users/pranoyr/Desktop/v_CricketShot_g11_c05.avi')
        cam = cv2.VideoCapture(
            './data/video_data/Suturing/Suturing_B001_capture1.avi'
        )
        clip = []
        #clip = clip.to(device)
        frame_count = 0
        grad_conter = 0
        pre = 0
        while True:
            ret, img = cam.read()
            if frame_count == 8:
                grad_conter = grad_conter + 1
                print(len(clip))
                ### 为了Grad-CAM绘图，多返回了一个heatmap
                preds, heatmap = predict(clip, model)
                plt.matshow(heatmap.squeeze())
                pass

                draw = img.copy()
                ### 融入Grad-CAM的绘画
                heatmap = np.array(heatmap)
                heatmap = cv2.resize(heatmap, (draw.shape[1], draw.shape[0]))
                heatmap = np.uint8(225 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + draw

                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    superimposed_img, idx_to_class[preds.item()], 
                    (100, 100), font, 1.0, (225, 225, 225), 
                    1, cv2.LINE_AA
                )
                cv2.imwrite(f'./grad_cam_pics/1_frame/grad_{grad_conter}.jpg', superimposed_img)

                # font = cv2.FONT_HERSHEY_SIMPLEX

                ### 我的魔改
                if preds.size(0) != 0:
                    print(idx_to_class[preds.item()])
                    cv2.putText(draw, idx_to_class[preds.item(
                    )], (100, 100), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
                    #print(draw == pre)
                    #pre = draw

                    cv2.imshow('window', draw)
                    cv2.waitKey(1)
                frame_count = 0
                clip = []

            #img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = Image.fromarray(img)
            clip.append(img)
            frame_count += 1