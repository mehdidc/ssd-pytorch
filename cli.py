import os
import numpy as np
from collections import defaultdict
from clize import run
import json
from skimage.io import imsave
import pandas as pd

import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.functional import smooth_l1_loss, mse_loss, cross_entropy
from torchvision import models
from model import SSD

from dataset import COCO
from dataset import SubSample

from util import encode_bounding_box_list
from util import encode_bounding_box_list_one_to_one
from util import encode_bounding_box_list_many_to_one
from util import decode_bounding_box_list
from util import build_anchors
from util import BOUNDING_BOX
from util import MASK
from util import CLASS_ID
from util import draw_bounding_boxes
from util import softmax
from util import non_maximal_suppression
from util import non_maximal_suppression_per_class
from util import precision
from util import recall

cudnn.benchmark = True

def train(*, folder='coco', resume=False, out_folder='out'):
    lambda_ = 1 # weight given to the classification loss relative to localization loss
    batch_size = 16
    num_epoch = 100000
    image_size = 300 
    lr = 0.001
    gamma = 0.9
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    imbalance_strategy = 'hard_negative_mining'
    pos_weight = 1
    neg_weight = 0.3

    normalize = transforms.Normalize(mean=mean, std=std)
 
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = COCO(folder, split='train2014', transform=train_transform)
    train_dataset = SubSample(train_dataset, nb=1000)
    valid_dataset = COCO(folder, split='val2014', transform=valid_transform)
    valid_dataset = SubSample(valid_dataset, nb=500)
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=lambda l:l,
        num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        collate_fn=lambda l:l,
        num_workers=4,
    )
    ar = [1, 2, 3, 1/2, 1/3]
    aspect_ratios = [ar] * 6
    anchor_list = [
        build_anchors(scale=0.2, feature_map_size=37, aspect_ratios=aspect_ratios[0]),   
        build_anchors(scale=0.34, feature_map_size=19, aspect_ratios=aspect_ratios[1]),   
        build_anchors(scale=0.48, feature_map_size=10, aspect_ratios=aspect_ratios[2]),   
        build_anchors(scale=0.62, feature_map_size=5, aspect_ratios=aspect_ratios[3]),   
        build_anchors(scale=0.76, feature_map_size=3, aspect_ratios=aspect_ratios[4]),   
        build_anchors(scale=0.90, feature_map_size=1, aspect_ratios=aspect_ratios[5]),   
    ]
    nb_classes = len(train_dataset.classes) + 1 # normal classes + background class
    nb_values_per_anchor = 4 + nb_classes # bounding box (4) + nb_classes
    
    print('Number of training images : {}'.format(len(train_dataset)))
    print('Number of valid images : {}'.format(len(valid_dataset)))
    print('Number of classes : {}'.format(nb_classes))
    
    if resume:
        model = torch.load('model.th')
        if os.path.exists('stats.csv'):
            stats = pd.read_csv('stats.csv').to_dict(orient='list')
            first_epoch = max(stats['epoch']) + 1
        else:
            stats = defaultdict(list)
            first_epoch = 0
    else:
        model = SSD(
            num_anchors=list(map(len, aspect_ratios)), 
            num_classes=nb_classes)
        model.transform = valid_transform
        model.nb_classes = nb_classes
        model.aspect_ratios = aspect_ratios
        model.nb_values_per_anchor = nb_values_per_anchor
        model.anchor_list = anchor_list
        model.image_size = image_size
        first_epoch = 0
        stats = defaultdict(list)
    
    class_weight = torch.zeros(nb_classes)
    class_weight[0] = neg_weight
    class_weight[1:] = pos_weight
    class_weight = class_weight.cuda()

    #cross_entropy = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model = model.cuda()

    avg_loss = 0.
    nb_updates = 0
    for epoch in range(first_epoch, num_epoch):
        model.train()
        for batch, samples, in enumerate(train_loader):
            X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred), masks = _predict(model, samples)
            # X is batch of image
            # Y is groundtruth output
            # Ypred is predicted output
            # bbox_true are groundtruth bounding boxes extracted from Y
            # bbox_pred are predicted bounding boxes extracted from Ypred
            # class_true are groundtruth classes extracted from Y
            # class_pred are predicted classes extracted from Ypred
            l_loc = 0
            l_classif = 0
            for i in range(len(Y)):
                m = masks[i]
                bt = bbox_true[i]
                bp = bbox_pred[i]
                ct = class_true[i]
                cp = class_pred[i]
                nb_pos = m.view(m.size(0), -1).sum(1).view(m.size(0), 1, 1, 1, 1)
                nb_neg = (1 - m).view(m.size(0), -1).sum(1).view(m.size(0), 1, 1, 1, 1)
                nb_examples = m.size(0)
                N = (nb_pos + 1) * (nb_examples)
                l_loc = ((m * smooth_l1_loss(bp, bt, size_average=False, reduce=False)) / N).sum() + l_loc
                
                if imbalance_strategy == 'hard_negative_mining':
                    # Hard negative mining
                    ind = torch.arange(len(ct))
                    pos = ind[(ct.data.cpu() > 0)].long().cuda()
                    neg = ind[(ct.data.cpu() == 0)].long().cuda()
                    
                    ct_pos = ct[pos]
                    cp_pos = cp[pos]
                    ct_neg = ct[neg]
                    cp_neg = cp[neg]
                    cp_neg_s = nn.Softmax(dim=1)(cp_neg)

                    vals, indices = cp_neg_s[:, 0].sort(descending=False)
                    nb = len(ct_pos) * 3
                    cp_neg = cp_neg[indices[0:nb]]
                    ct_neg = ct_neg[indices[0:nb]]
                    l_classif = cross_entropy(cp_pos, ct_pos) + cross_entropy(cp_neg, ct_neg) + l_classif
                elif imbalance_strategy == 'class_weight':
                    l_classif = cross_entropy(cp, ct, weight=class_weight) + l_classif
                elif imbalance_strategy == 'nothing':
                    l_classif = cross_entropy(cp, ct) + l_classif

            model.zero_grad()
            loss = l_loc + lambda_ * l_classif
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            print('Batch {:05d}/{:05d} AvgTrainLoss : {:.4f}'.format(batch, len(train_loader), avg_loss))
            if nb_updates % 100 == 0:
                torch.save(model, 'model.th')
                X = X.data.cpu().numpy()
                Y = [y.data.cpu().numpy() for y in Y]
                Ypred = [ypr.data.cpu().numpy() for ypr in Ypred]
                for i in range(len(X)):
                    x = X[i]
                    x = x.transpose((1, 2, 0))
                    x = x * np.array(std) + np.array(mean)
                    x = x.astype('float32')
                    gt_boxes = []
                    pred_boxes = []
                    for j in range(len(Y)):
                        y = Y[j][i, :, 0:5]#x,y,w,h,cl
                        ypred = Ypred[j][i, :]#x,y,w,h,cl0_score,cl1_score,cl2_score,...
                        A = model.anchor_list[j]
                        boxes = decode_bounding_box_list(y, A, include_scores=False)
                        gt_boxes.extend(boxes)
                        boxes = decode_bounding_box_list(ypred, A, include_scores=True)
                        pred_boxes.extend(boxes)
                    pred_boxes = non_maximal_suppression_per_class(pred_boxes)
                    gt_boxes = [(box, train_dataset.class_name[class_id]) for box, class_id in gt_boxes]
                    pred_boxes = [(box, train_dataset.class_name[class_id]) for box, class_id in pred_boxes]
                    x = draw_bounding_boxes(x, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0))
                    x = draw_bounding_boxes(x, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0)) 
                    imsave(os.path.join(out_folder, 'sample_{:05d}.jpg'.format(i)), x)
            nb_updates += 1
        model.eval()
        metrics = defaultdict(list)
        for split_name, loader in (('train', train_loader), ('valid', valid_loader)):
            for batch, samples, in enumerate(loader):
                X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred), masks = _predict(model, samples)
                Y = [y.data.cpu().numpy() for y in Y]
                Ypred = [ypr.data.cpu().numpy() for ypr in Ypred]
                for i in range(len(X)):
                    gt_boxes = samples[i][1]
                    pred_boxes = []
                    for j in range(len(Y)):
                        ypred = Ypred[j][i, :]
                        A = model.anchor_list[j]
                        boxes = decode_bounding_box_list(ypred, A, include_scores=True)
                        pred_boxes.extend(boxes)
                    pred_boxes = non_maximal_suppression_per_class(pred_boxes)
                    prec = precision(pred_boxes, gt_boxes)
                    re = recall(pred_boxes, gt_boxes)
                    metrics['precision_' + split_name].append(prec)
                    metrics['recall_' + split_name].append(re)
        for k in sorted(metrics.keys()):
            v = np.mean(metrics[k])
            print('{}: {:.4}'.format(k, v))
            stats[k].append(v)
        stats['epoch'].append(epoch)
        pd.DataFrame(stats).to_csv('stats.csv', index=False)


def _predict(model, samples):
    X = torch.stack([x for x, _ in samples], 0)
    bboxes = [y for _, y in samples]
    
    X = X.cuda()
    X = Variable(X)
     
    Y = [[encode_bounding_box_list_one_to_one(bbox, A) for bbox in bboxes] for A in model.anchor_list]
    Y = [Variable(torch.from_numpy(np.array(y)).float().cuda()) for y in Y]
    # X has shape (nb_examples, 3, image_size, image_size)
    # each element of Y is a scale and has shape (nb_examples, nb_anchors_per_position, 6, feature_map_size, feature_map_size)
    
    Ypred = model(X)
    Ypred = [ypr.view(ypr.size(0), len(model.aspect_ratios[i]), model.nb_values_per_anchor, ypr.size(2), ypr.size(3))     
             for i, ypr in enumerate(Ypred)]
    # each element of Ypred has shape (nb_examples, nb_anchors_per_position, nb_values_per_anchor, feature_map_size, feature_map_size)
    
    bbox_true = [y[:, :, BOUNDING_BOX] for y in Y]
    # each element of bbox_true has shape (nb_examples, nb_anchors_per_position, 4, feature_map_size, feature_map_size)
    
    masks = [y[:, :, MASK:MASK + 1].contiguous() for y in Y]
    # each element of mask has shape (nb_examples, nb_anchors_per_position, 1, feature_map_size, feature_map_size)
    
    class_true = [m * y[:, :, CLASS_ID:CLASS_ID+1] for m, y in zip(masks, Y)]
    class_true = [c.contiguous().view(-1).long() for c in class_true]
    # each element of class_true has shape (nb_examples * nb_anchors_per_position * feature_map_size * feature_map_size)
    
    bbox_pred = [ypr[:, :, BOUNDING_BOX] for ypr in Ypred] 
    # each element of bbox_pred has shape (nb_examples, nb_anchors_per_position, 4, feature_map_size, feature_map_size)

    class_pred = [ypr[:, :, len(BOUNDING_BOX):] for ypr in Ypred] 
    # each element of class_pred has shape (nb_examples, nb_anchors_per_position, nb_classes, feature_map_size, feature_map_size)
    class_pred = [cp.permute(0, 1, 3, 4, 2).contiguous().view(-1, model.nb_classes).contiguous() for cp in class_pred]
    # each element of class_pred has shape (-1, model.nb_classes)

    return X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred), masks

if __name__ == '__main__':
    run([train])
