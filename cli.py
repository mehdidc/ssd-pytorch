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
from dataset import VOC
from dataset import SubSample

from optim import CyclicLR

import pyximport; pyximport.install()
from bounding_box import decode_bounding_box_list
from bounding_box import non_maximal_suppression_per_class
from bounding_box import precision
from bounding_box import recall
from bounding_box import build_anchors
from bounding_box import softmax
from bounding_box import draw_bounding_boxes
from bounding_box import average_precision
cudnn.benchmark = True

def train(*, config='config', resume=False):
    cfg = {}
    exec(open(config).read(), cfg, cfg)
     
    lambda_ = cfg['lambda_'] # given to the classification loss relative to localization loss
    batch_size = cfg['batch_size']
    num_epoch = cfg['num_epoch']
    image_size = cfg['image_size']
    lr = cfg['lr']
    gamma = cfg['gamma']
    mean = cfg['mean']
    std = cfg['std']
    dataset = cfg['dataset']
    imbalance_strategy = cfg['imbalance_strategy']
    out_folder = cfg['out_folder'] 
    classes = cfg.get('classes')
    if imbalance_strategy == 'class_weight':
        pos_weight = 1
        neg_weight = 0.3
    nms_iou_threshold = cfg['nms_iou_threshold']
    bbox_encoding_iou_threshold = cfg['bbox_encoding_iou_threshold']
    eval_iou_threshold = cfg['eval_iou_threshold']
    aspect_ratios = cfg['aspect_ratios']
    log_interval = cfg.get('log_interval', 100)
    eval_topk = cfg['eval_topk'] 
    score_threshold = cfg['score_threshold']

    debug = False 
    if debug:
        log_interval = 30
    # anchor list for each scale (we have 6 scales)
    # (order is important)
    scales = cfg['scales']
    anchor_list = [
        build_anchors(scale=scales[0], feature_map_size=37, aspect_ratios=aspect_ratios[0]),   
        build_anchors(scale=scales[1], feature_map_size=19, aspect_ratios=aspect_ratios[1]),   
        build_anchors(scale=scales[2], feature_map_size=10, aspect_ratios=aspect_ratios[2]),   
        build_anchors(scale=scales[3], feature_map_size=5, aspect_ratios=aspect_ratios[3]),   
        build_anchors(scale=scales[4], feature_map_size=3, aspect_ratios=aspect_ratios[4]),   
        build_anchors(scale=scales[5], feature_map_size=1, aspect_ratios=aspect_ratios[5]),   
    ]
    # transforms
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
    # dataset for train and valid
    print('Loading dataset anotations...')
    if dataset == 'COCO':
        train_split = 'train' + str(cfg['dataset_version'])
        val_split = 'val' + str(cfg['dataset_version'])
        kwargs = dict(
            folder=cfg['dataset_root_folder'],
            anchor_list=anchor_list,
            iou_threshold=bbox_encoding_iou_threshold,
            classes=classes,
            transform=train_transform
        )
        train_dataset = COCO(
            split=train_split,
            data_augmentation_params=cfg['data_augmentation_params'],
            **kwargs
        )
        valid_dataset = COCO(
            split=val_split,
            data_augmentation_params={},
            **kwargs,
        )
        train_evaluation = SubSample(COCO(
            split=train_split,
            data_augmentation_params={},
            **kwargs,
        ), nb=cfg['train_evaluation_size'])
        valid_evaluation = SubSample(
            valid_dataset, 
            nb=cfg['val_evaluation_size']
        )
    elif dataset == 'VOC':
        kwargs = dict(
            folder=cfg['dataset_root_folder'],
            anchor_list=anchor_list, 
            which=cfg['dataset_version'], 
            split='train', 
            iou_threshold=bbox_encoding_iou_threshold,
            classes=classes,
            transform=train_transform
        )
        train_dataset = VOC(
            split='train',
            data_augmentation_params=cfg['data_augmentation_params'],
            **kwargs,
        )
        valid_dataset = VOC(
            split='val',
            data_augmentation_params={},
            **kwargs,
        )
        train_evaluation = SubSample(VOC(
            split='train',
            data_augmentation_params={},
            **kwargs
        ), nb=cfg['train_evaluation_size'])
        valid_evaluation = SubSample(
            valid_dataset, 
            nb=cfg['val_evaluation_size']
        )
    else:
        raise ValueError('Unknown dataset {}'.format(dataset))
    print('Done loading dataset annotations.')
    if debug:
        n = 2
        train_dataset = SubSample(train_dataset, nb=n)
        valid_dataset = SubSample(valid_dataset, nb=n)
        train_evaluation = SubSample(train_evaluation, nb=n)
        valid_evaluation = SubSample(valid_evaluation, nb=n)
    assert train_dataset.class_to_idx == valid_dataset.class_to_idx
    assert train_dataset.idx_to_class == valid_dataset.idx_to_class
    clfn = lambda l:l
    # Dataset loaders for full training and full validation 
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=clfn,
        num_workers=cfg.get('num_workers', 8),
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        collate_fn=clfn,
        num_workers=cfg.get('num_workers', 8),
    )
    train_evaluation_loader = DataLoader(
        train_evaluation,
        batch_size=batch_size,
        collate_fn=clfn,
        num_workers=cfg.get('num_workers', 8)
    )
    valid_evaluation_loader = DataLoader(
        valid_evaluation,
        batch_size=batch_size,
        collate_fn=clfn,
        num_workers=cfg.get('num_workers', 8)
    )
    nb_classes = len(train_dataset.class_to_idx)
    bgid = train_dataset.class_to_idx['background']
    print('Number of training images : {}'.format(len(train_dataset)))
    print('Number of valid images : {}'.format(len(valid_dataset)))
    print('Number of classes : {}'.format(nb_classes))
    stats_filename = os.path.join(out_folder, 'stats.csv')
    train_stats_filename = os.path.join(out_folder, 'train_stats.csv')
    model_filename = os.path.join(out_folder, 'model.th')

    if resume:
        model = torch.load(model_filename)
        model = model.cuda()
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
    else:
        model = SSD(
            num_anchors=list(map(len, aspect_ratios)), 
            num_classes=nb_classes)
        model = model.cuda()
        model.transform = valid_transform
        model.nb_classes = nb_classes
        model.aspect_ratios = aspect_ratios
        model.anchor_list = anchor_list
        model.image_size = image_size
        first_epoch = 0
        stats = defaultdict(list)
        train_stats = defaultdict(list)
        model.nb_updates = 0
        model.avg_loss = 0.
        model.avg_loc = 0.
        model.avg_classif = 0
        model.background_class_id = train_dataset.background_class_id
        model.class_to_idx = train_dataset.class_to_idx
        model.mean = mean
        model.std = std
        print(model)

    if imbalance_strategy == 'class_weight':
        class_weight = torch.zeros(nb_classes)
        class_weight[0] = neg_weight
        class_weight[1:] = pos_weight
        class_weight = class_weight.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    step_size = len(train_dataset) // batch_size
    #scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=6e-3, step_size=step_size * 2)
    for epoch in range(first_epoch, num_epoch):
        epoch_t0 = time.time()
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
            N = max(nb_pos.data[0], 1.0)
            # localization loss
            l_loc = smooth_l1_loss(bp[ind], bt[ind], size_average=False) / N
            # classif loss
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
                nb = len(ct_pos) * 3 # 3x more neg than pos as in the paper
                cp_neg = cp_neg[indices[0:nb]]
                ct_neg = ct_neg[indices[0:nb]]
                l_classif = (cross_entropy(cp_pos, ct_pos, size_average=False) + cross_entropy(cp_neg, ct_neg, size_average=False)) / N
            elif imbalance_strategy == 'class_weight':
                l_classif = cross_entropy(cp, ct, weight=class_weight, size_average=False) / N
            elif imbalance_strategy == 'nothing':
                l_classif = cross_entropy(cp, ct, size_average=False) / N
            model.zero_grad()
            loss = l_loc + lambda_ * l_classif
            loss.backward()
            #scheduler.batch_step()
            optimizer.step()
            model.avg_loss = model.avg_loss * gamma + loss.data[0] * (1 - gamma)
            model.avg_loc = model.avg_loc * gamma  + l_loc.data[0] * (1 - gamma)
            model.avg_classif = model.avg_classif * gamma + l_classif.data[0] * (1 - gamma)
            delta = time.time() - t0
            print('Epoch {:05d}/{:05d} Batch {:05d}/{:05d} Loss : {:.3f} Loc : {:.3f} '
                  'Classif : {:.3f} AvgTrainLoss : {:.3f} AvgLoc : {:.3f} '
                  'AvgClassif {:.3f} Time:{:.3f}s'.format(
                      epoch + 1,
                      num_epoch,
                      batch + 1, 
                      len(train_loader), 
                      loss.data[0], 
                      l_loc.data[0],
                      l_classif.data[0],
                      model.avg_loss,
                      model.avg_loc,
                      model.avg_classif,
                      delta
                    ))
            train_stats['loss'].append(loss.data[0])
            train_stats['loc'].append(l_loc.data[0])
            train_stats['classif'].append(l_classif.data[0])
            train_stats['time'].append(delta)

            if model.nb_updates % log_interval == 0:
                # reporting part
                # -- draw training samples with their predicted and true bounding boxes
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
                # B contains groundtruth bounding boxes for each scale
                # C contains groundtruth classes for each scale
                # M contains the mask for each scale
                BP = [
                    bp.data.cpu().view(bp.size(0), -1, 4, bp.size(2), bp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
                CP = [
                    cp.data.cpu().view(cp.size(0), -1, model.nb_classes, cp.size(2), cp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
                # BP contains predicted bounding boxes for each scale
                # CP contains predicted classes for each scale

                for i in range(len(X)):
                    # for each example i in mini-batch
                    x = X[i]
                    x = x.transpose((1, 2, 0))
                    x = x * np.array(std) + np.array(mean)
                    x = x.astype('float32')
                    gt_boxes = []
                    pred_boxes = []
                    for j in range(len(Y)):
                        # for each scale j
                        ct = C[j][i]# class_id
                        cp = CP[j][i]#cl1_score,cl2_score,...
                        bt = B[j][i]#4
                        bp = BP[j][i]#4
                        A = model.anchor_list[j]
                        # get groundtruth boxes
                        gt_boxes.extend(decode_bounding_box_list(
                            bt, ct, A, 
                            include_scores=False,
                            image_size=image_size,
                        ))
                        # get predicted boxes
                        pred_boxes.extend(decode_bounding_box_list(
                            bp, cp, A, 
                            include_scores=True,
                            image_size=image_size,
                        ))
                    gt_boxes = [(box, class_id) for box, class_id in gt_boxes if class_id != bgid]
                    # apply non-maximal suppression to predicted boxes
                    pred_boxes = non_maximal_suppression_per_class(pred_boxes, score_threshold=score_threshold)
                    pred_boxes = [(box, class_id) for box, class_id in pred_boxes if class_id != bgid]
                    # get class names
                    gt_boxes = [(box, train_dataset.idx_to_class[class_id]) for box, class_id in gt_boxes]
                    pred_boxes = [(box, train_dataset.idx_to_class[class_id]) for box, class_id in pred_boxes]
                    # draw boxes
                    pad = 30
                    im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
                    im[pad:-pad, pad:-pad] = x
                    im = draw_bounding_boxes(im, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0), pad=pad)
                    im = draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
                    imsave(os.path.join(out_folder, 'sample_{:05d}.jpg'.format(i)), im)
                delta = time.time() - t0
                print('Draw box time {:.4f}s'.format(delta))
            model.nb_updates += 1
            # LR schedule
            _update_lr(optimizer, model.nb_updates, cfg['lr_schedule'])
        if debug and model.nb_updates % 50 != 0:
            continue
        epoch_time = time.time() - epoch_t0
        print('Evaluation') 
        t0 = time.time()
        model.eval()
        metrics = defaultdict(list)
        metrics['train_time'] = [epoch_time]
        for split_name, loader in (('train', train_evaluation_loader), ('valid', valid_evaluation_loader)):
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
                 
                # B contains groundtruth bounding boxes for each scale
                # C contains groundtruth classes for each scale
                # M contains the mask for each scale

                BP = [
                    bp.data.cpu().view(bp.size(0), -1, 4, bp.size(2), bp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
                CP = [
                    cp.data.cpu().view(cp.size(0), -1, model.nb_classes, cp.size(2), cp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]

                # BP contains predicted bounding boxes for each scale
                # CP contains predicted classes for each scale

                for i in range(len(X)):
                    # for each example i  in mini-batch
                    gt_boxes = []
                    pred_boxes = []
                    for j in range(len(Y)):
                        # for each scale j
                        ct = C[j][i]# class_id
                        cp = CP[j][i]#cl1_score,cl2_score,...
                        bt = B[j][i]#4
                        bp = BP[j][i]#4
                        A = model.anchor_list[j]
                        # get groundtruth boxes
                        gt_boxes.extend(decode_bounding_box_list(
                            bt, ct, A, 
                            include_scores=False,
                            image_size=image_size,
                        ))
                        # get predicted boxes
                        pred_boxes.extend(decode_bounding_box_list(
                            bp, cp, A, 
                            include_scores=True,
                            image_size=image_size,
                        ))
                    gt_boxes = [(box, class_id) for box, class_id in gt_boxes if class_id != bgid]
                    # mAP
                    mAP = []
                    for class_id in model.class_to_idx.values():
                        if class_id == bgid:
                            continue
                        AP = average_precision(pred_boxes, gt_boxes, class_id, topk=eval_topk, iou_threshold=eval_iou_threshold)
                        if AP is not None:
                            mAP.append(AP)
                    mAP = np.mean(mAP)
                    metrics['mAP_' + split_name].append(mAP)
                    # use the predicted boxes and groundtruth boxes to compute precision and recall
                    pred_boxes = non_maximal_suppression_per_class(pred_boxes, score_threshold=score_threshold)
                    P = []
                    R = []
                    for class_id in model.class_to_idx.values():
                        if class_id == bgid:
                            continue
                        t = [box for box, cl in gt_boxes if cl == class_id]
                        if len(t) == 0:
                            continue
                        p = [box for box, cl in pred_boxes if cl == class_id]
                        prec = precision(p, t, iou_threshold=eval_iou_threshold)
                        re = recall(p, t, iou_threshold=eval_iou_threshold)
                        P.append(prec)
                        R.append(re)
                    metrics['precision_' + split_name].append(np.mean(P))
                    metrics['recall_' + split_name].append(np.mean(R))
                delta = time.time() - tt0
                print('Eval Batch {:04d}/{:04d} on split {} Time : {:.3f}s'.format(batch, len(loader), split_name, delta))
            delta = time.time() - t0
            metrics['eval_' + split_name + '_time'] = [delta]
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


def _update_lr(optimizer, nb_iter, schedule):
    for sc in schedule:
        (start_iter, end_iter), new_lr = sc['iter'], sc['lr']
        if start_iter <= nb_iter  <= end_iter:
            break
    old_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        print('Chaning LR from {:.5f} to {:.5f}'.format(old_lr, new_lr))
    for g in optimizer.param_groups:
        g['lr'] = new_lr


def test(filename, *, model='out/model.th', out=None, cuda=False):
    if not out:
        path, ext = filename.split('.', 2)
        out = path + '_out' + '.' + ext
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
        boxes = decode_bounding_box_list(
            bp, cp, A, 
            image_size=model.image_size,
            include_scores=True,
        )
        pred_boxes.extend(boxes)
    pred_boxes = non_maximal_suppression_per_class(pred_boxes, score_threshold=score_threshold)
    bgid = model.class_to_idx['background']
    pred_boxes = [(box, class_id) for box, class_id in pred_boxes if class_id != bgid]
    pred_boxes = [(box, model.idx_to_class[class_id]) for box, class_id in pred_boxes]
    for box, name in pred_boxes:
        print(name)
    pad = 30
    im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
    im[pad:-pad, pad:-pad] = x
    im = draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
    imsave(out, x)


if __name__ == '__main__':
    run([train, test])
