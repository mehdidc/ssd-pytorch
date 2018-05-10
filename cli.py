import time
import sys
import os
import numpy as np
from collections import defaultdict
from clize import run
import json
from skimage.io import imsave
import pandas as pd
from joblib import Parallel, delayed

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
from torchvision.datasets.folder import default_loader

from dataset import COCO
from dataset import SubSample

from util import draw_bounding_boxes
from util import softmax
from optim import CyclicLR

import pyximport; pyximport.install()
from bounding_box import encode_bounding_box_list_many_to_one
from bounding_box import encode_bounding_box_list_one_to_one
from bounding_box import decode_bounding_box_list
from bounding_box import non_maximal_suppression
from bounding_box import non_maximal_suppression_per_class
from bounding_box import precision
from bounding_box import recall
from bounding_box import build_anchors


cudnn.benchmark = True


def train(*, folder='coco', resume=False, out_folder='out'):
    lambda_ = 1 # given to the classification loss relative to localization loss
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

    train_dataset = COCO(folder, anchor_list, split='train2014', transform=train_transform)
    #train_dataset = SubSample(train_dataset, nb=16)
    valid_dataset = COCO(folder, anchor_list, split='val2014', transform=valid_transform)
    #valid_dataset = SubSample(valid_dataset, nb=16)
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=lambda l:l,
        num_workers=8,
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        collate_fn=lambda l:l,
        num_workers=8,
    )
    

    train_subset = SubSample(train_dataset, nb=1000)
    valid_subset = SubSample(valid_dataset, nb=1000)
    
    train_loader_subset = DataLoader(
        train_subset,
        batch_size=batch_size,
        collate_fn=lambda l:l,
        num_workers=8
    )

    valid_loader_subset = DataLoader(
        valid_subset,
        batch_size=batch_size,
        collate_fn=lambda l:l,
        num_workers=8
    )

    nb_classes = len(train_dataset.classes) + 1 # normal classes + background class
    nb_values_per_anchor = 4 + nb_classes # bounding box (4) + nb_classes
    
    print('Number of training images : {}'.format(len(train_dataset)))
    print('Number of valid images : {}'.format(len(valid_dataset)))
    print('Number of classes : {}'.format(nb_classes))
    
    stats_filename = os.path.join(out_folder, 'stats.csv')
    train_stats_filename = os.path.join(out_folder, 'train_stats.csv')
    model_filename = os.path.join(out_folder, 'model.th')

    if resume:
        model = torch.load(model_filename)

        if os.path.exists(stats_filename):
            stats = pd.read_csv(stats_filename).to_dict(orient='list')
            first_epoch = max(stats['epoch']) + 1
        else:
            stats = defaultdict(list)
            first_epoch = 0

        if os.path.exists(train_stats_filename):
            train_stats = pd.read_csv(train_stats_filename).to_dict(orient='list')
        else:
            train_stats = defaultdict(list)

        if not hasattr(model, 'nb_updates'):
            model.nb_updates = 0

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
        train_stats = defaultdict(list)
        model.nb_updates = 0

    model.class_name = train_dataset.class_name
    model.mean = mean
    model.std = std
    
    class_weight = torch.zeros(nb_classes)
    class_weight[0] = neg_weight
    class_weight[1:] = pos_weight
    class_weight = class_weight.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    step_size = len(train_dataset) // batch_size
    scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=6e-3, step_size=step_size * 2)
    model = model.cuda()

    avg_loss = 0.
    avg_loc = 0.
    avg_classif = 0.
    for epoch in range(first_epoch, num_epoch):
        model.train()
        for batch, samples, in enumerate(train_loader):
            t0 = time.time()
            X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred), masks = _predict(model, samples)
            # X is batch of image
            # Y is groundtruth output
            # Ypred is predicted output
            # bbox_true are groundtruth bounding boxes extracted from Y
            # bbox_pred are predicted bounding boxes extracted from Ypred
            # class_true are groundtruth classes extracted from Y
            # class_pred are predicted classes extracted from Ypred
            m = masks.view(-1)
            ind = m.nonzero().view(-1)
            bt = bbox_true
            bp = bbox_pred
            ct = class_true
            cp = class_pred
            nb_pos = m.long().sum()
            #nb_examples = m.size(0)
            #N = (nb_pos + 1) * (nb_examples)
            #l_loc = ((m * smooth_l1_loss(bp, bt, size_average=False, reduce=False)) / N).sum()
            l_loc = smooth_l1_loss(bp[ind], bt[ind])
            if imbalance_strategy == 'hard_negative_mining':
                # Hard negative mining
                ind = torch.arange(len(ct))
                pos = ind[(ct.data.cpu() > 0)].long().cuda()
                neg = ind[(ct.data.cpu() == 0)].long().cuda()
                ct_pos = ct[pos]
                cp_pos = cp[pos]
                ct_neg = ct[neg]
                cp_neg = cp[neg]
                cp_neg_loss = cross_entropy(cp_neg, ct_neg, reduce=False)
                vals, indices = cp_neg_loss.sort(descending=True)
                nb = len(ct_pos) * 3
                cp_neg = cp_neg[indices[0:nb]]
                ct_neg = ct_neg[indices[0:nb]]
                l_classif = cross_entropy(cp_pos, ct_pos) + cross_entropy(cp_neg, ct_neg)
            elif imbalance_strategy == 'class_weight':
                l_classif = cross_entropy(cp, ct, weight=class_weight)
            elif imbalance_strategy == 'nothing':
                l_classif = cross_entropy(cp, ct)
            model.zero_grad()
            loss = l_loc + lambda_ * l_classif
            loss.backward()
            scheduler.batch_step()
            optimizer.step()
            if avg_loss == 0.0:
                avg_loss = loss.data[0]
                avg_loc = l_loc.data[0]
                avg_classif = l_classif.data[0]
            else:
                avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
                avg_loc = avg_loc * gamma  + l_loc.data[0] * (1 - gamma)
                avg_classif = avg_classif * gamma + l_classif.data[0] * (1 - gamma)
            delta = time.time() - t0
            print('Epoch {:05d}/{:05d} Batch {:05d}/{:05d} Loss : {:.3f} Loc : {:.3f} '
                  'Classif : {:.3f} AvgTrainLoss : {:.3f} AvgLoc : {:.3f} AvgClassif {:.3f} Time:{:.3f}s'.format(
                epoch,
                num_epoch,
                batch, 
                len(train_loader), 
                loss.data[0], 
                l_loc.data[0],
                l_classif.data[0],
                avg_loss,
                avg_loc,
                avg_classif,
                delta
                ))
            train_stats['avg_loss'].append(avg_loss)
            train_stats['loss'].append(loss.data[0])
            train_stats['avg_loc'].append(avg_loc)
            train_stats['loc'].append(l_loc.data[0])
            train_stats['avg_classif'].append(avg_classif)
            train_stats['classif'].append(l_classif.data[0])

            if model.nb_updates % 100 == 0:
                pd.DataFrame(train_stats).to_csv(train_stats_filename, index=False)
                t0 = time.time()
                torch.save(model, model_filename)
                X = X.data.cpu().numpy()
                
                B = [[b for b, c, m in y] for y in Y]
                B = [torch.from_numpy(np.array(b)).float() for b in B]
                C = [[c for b, c, m in y] for y in Y]
                C = [torch.from_numpy(np.array(c)).long() for c in C]
                M = [[m for b, c, m in y] for y in Y]
                M = [torch.from_numpy(np.array(m).astype('uint8')) for m in M]
 
                BP = [
                    bp.data.cpu().view(bp.size(0), -1, 4, bp.size(2), bp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
                CP = [
                    cp.data.cpu().view(cp.size(0), -1, model.nb_classes, cp.size(2), cp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
 
                for i in range(len(X)):
                    x = X[i]
                    x = x.transpose((1, 2, 0))
                    x = x * np.array(std) + np.array(mean)
                    x = x.astype('float32')
                    gt_boxes = []
                    pred_boxes = []
                    for j in range(len(Y)):
                        ct = C[j][i]# class_id
                        cp = CP[j][i]#cl1_score,cl2_score,...
                        bt = B[j][i]#4
                        bp = BP[j][i]#4
                        A = model.anchor_list[j]
                        boxes = decode_bounding_box_list(bt, ct, A, include_scores=False)
                        gt_boxes.extend(boxes)
                        boxes = decode_bounding_box_list(bp, cp, A, include_scores=True)
                        pred_boxes.extend(boxes)
                    pred_boxes = non_maximal_suppression_per_class(pred_boxes)
                    gt_boxes = [(box, train_dataset.class_name[class_id]) for box, class_id in gt_boxes]
                    pred_boxes = [(box, train_dataset.class_name[class_id]) for box, class_id in pred_boxes]
                    x = draw_bounding_boxes(x, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0))
                    x = draw_bounding_boxes(x, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0)) 
                    imsave(os.path.join(out_folder, 'sample_{:05d}.jpg'.format(i)), x)
                delta = time.time() - t0
                print('Draw box time {:.4f}s'.format(delta))
            model.nb_updates += 1
        print('Evaluation') 
        t0 = time.time()
        model.eval()
        metrics = defaultdict(list)
        for split_name, loader in (('train', train_loader_subset), ('valid', valid_loader_subset)):
            t0 = time.time()
            for batch, samples, in enumerate(loader):
                tt0 = time.time()
                X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred), masks = _predict(model, samples)
                B = [[b for b, c, m in y] for y in Y]
                B = [torch.from_numpy(np.array(b)).float() for b in B]
                C = [[c for b, c, m in y] for y in Y]
                C = [torch.from_numpy(np.array(c)).long() for c in C]
                M = [[m for b, c, m in y] for y in Y]
                M = [torch.from_numpy(np.array(m).astype('uint8')) for m in M]
 
                BP = [
                    bp.data.cpu().view(bp.size(0), -1, 4, bp.size(2), bp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
                CP = [
                    cp.data.cpu().view(cp.size(0), -1, model.nb_classes, cp.size(2), cp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
 
                for i in range(len(X)):
                    _, gt_boxes, _ = samples[i]
                    pred_boxes = []
                    for j in range(len(Y)):
                        cp = CP[j][i]#cl1_score,cl2_score,...
                        bp = BP[j][i]#4
                        A = model.anchor_list[j]
                        boxes = decode_bounding_box_list(bp, cp, A, include_scores=True)
                        pred_boxes.extend(boxes)
                    pred_boxes = non_maximal_suppression_per_class(pred_boxes)
                    prec = precision(pred_boxes, gt_boxes)
                    re = recall(pred_boxes, gt_boxes)
                    metrics['precision_' + split_name].append(prec)
                    metrics['recall_' + split_name].append(re)
                delta = time.time() - tt0
                print('Eval Batch {:04d}/{:04d} on split {} Time : {:.3f}s'.format(batch, len(loader), split_name, delta))
            delta = time.time() - t0
            print('Eval time of {}: {:.4f}s'.format(split_name, delta))

        for k in sorted(metrics.keys()):
            v = np.mean(metrics[k])
            print('{}: {:.4}'.format(k, v))
            stats[k].append(v)
        stats['epoch'].append(epoch)
        pd.DataFrame(stats).to_csv(stats_filename, index=False)


def _predict(model, samples):
    X = torch.stack([x for x, _, _ in samples], 0) 
    X = X.cuda()
    X = Variable(X)
    # X has shape (nb_examples, 3, image_size, image_size)
    bbox_encodings = [be for _, _, be in samples]
    Y = list(zip(*bbox_encodings))
    
    B = [[b for b, c, m in y] for y in Y]
    B = [torch.from_numpy(np.array(b)).float() for b in B]
    C = [[c for b, c, m in y] for y in Y]
    C = [torch.from_numpy(np.array(c)).long() for c in C]
    M = [[m for b, c, m in y] for y in Y]
    M = [torch.from_numpy(np.array(m).astype('uint8')) for m in M]
    bt = [b.view(-1, 4) for b in B]
    bt = torch.cat(bt, 0)
    # B has shape (*, 6)

    ct = [c.view(-1) for c in C]
    ct = torch.cat(ct, 0)
    
    M = [m.view(-1) for m in M]
    M = torch.cat(M, 0)

    Ypred = model(X)
    bp = [b.view(b.size(0), -1, 4, b.size(2), b.size(3)).permute(0, 3, 4, 1, 2).contiguous() for b, c in Ypred]
    cp = [c.view(c.size(0), -1, model.nb_classes, c.size(2), c.size(3)).permute(0, 3, 4, 1, 2).contiguous() for b, c in Ypred]    
    
    bp = [b.view(-1, 4) for b in bp]
    bp = torch.cat(bp, 0)
    
    cp = [c.view(-1, model.nb_classes) for c in cp]
    cp = torch.cat(cp, 0)

    ct = Variable(ct).cuda()
    bt = Variable(bt).cuda()
    M = Variable(M).cuda()
    return X, (Y, Ypred), (bt, bp), (ct, cp), M


def test(filename, *, model='out/model.th', out='out.png', cuda=False):
    model = torch.load(model, map_location=lambda storage, loc: storage)
    if cuda:
        model = model.cuda()
    model.eval()
    im = default_loader(filename)
    x = model.transform(im)
    X = x.view(1, x.size(0), x.size(1), x.size(2))
    if cuda:
        X = X.cuda()
    X = Variable(X)
    Ypred = model(X)
    BP = [
        bp.data.cpu().
        view(bp.size(0), -1, 4, bp.size(2), bp.size(3)).
        permute(0, 3, 4, 1, 2).
        numpy() 
    for bp, cp in Ypred]
    CP = [
        cp.data.cpu().
        view(cp.size(0), -1, model.nb_classes, cp.size(2), cp.size(3)).
        permute(0, 3, 4, 1, 2).
        numpy() 
    for bp, cp in Ypred]

    X = X.data.cpu().numpy()
    x = X[0]
    x = x.transpose((1, 2, 0))
    x = x * np.array(model.std) + np.array(model.mean)
    x = x.astype('float32')
    pred_boxes = []
    for j in range(len(Ypred)):
        cp = CP[j][0]#cl1_score,cl2_score,...
        bp = BP[j][0]#4
        A = model.anchor_list[j]
        boxes = decode_bounding_box_list(bp, cp, A, include_scores=True)
        pred_boxes.extend(boxes)
    pred_boxes = sorted(pred_boxes, key=lambda p:p[2], reverse=True)
    #pred_boxes = [(box, class_id, score) for (box, class_id, score) in pred_boxes if score > 0.1]
    pred_boxes = non_maximal_suppression_per_class(pred_boxes)
    pred_boxes = [(box, model.class_name[class_id]) for box, class_id in pred_boxes]
    for box, name in pred_boxes:
        print(name)
    x = draw_bounding_boxes(x, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0)) 
    imsave(out, x)

if __name__ == '__main__':
    run([train, test])
