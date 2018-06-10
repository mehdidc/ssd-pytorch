import time
import os
import numpy as np
from collections import defaultdict
from clize import run
from skimage.io import imsave
import pandas as pd

import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.functional import smooth_l1_loss, cross_entropy
import model as model_module
from torchvision.datasets.folder import default_loader

from dataset import COCO
from dataset import VOC
from dataset import SubSample


import pyximport; pyximport.install()
from bounding_box import decode_bounding_box_list
from bounding_box import non_maximal_suppression_per_class
from bounding_box import precision
from bounding_box import recall
from bounding_box import build_anchors
from bounding_box import draw_bounding_boxes
from bounding_box import average_precision

cudnn.benchmark = True


def train(*, config='config', resume=False):
    cfg = _read_config(config)
    w_loc = cfg['w_loc']
    w_classif = cfg['w_classif']
    batch_size = cfg['batch_size']
    num_epoch = cfg['num_epoch']
    image_size = cfg['image_size']
    gamma = cfg['gamma']
    mean = cfg['mean']
    std = cfg['std']
    imbalance_strategy = cfg['imbalance_strategy']
    out_folder = cfg['out_folder'] 
    background_ratio = cfg['background_ratio']
    if imbalance_strategy == 'class_weight':
        pos_weight = 1
        neg_weight = 0.01
    nms_iou_threshold = cfg['nms_iou_threshold']
    eval_iou_threshold = cfg['eval_iou_threshold']
    log_interval = cfg['log_interval']
    nms_score_threshold = cfg['nms_score_threshold']
    eval_interval = cfg['eval_interval']
    aspect_ratios = cfg['aspect_ratios']
    nms_topk = cfg['nms_topk']
    debug = cfg['debug'] 

    folders = [
        'train', 
        'eval_train', 
        'eval_valid', 
    ]
    for f in folders:
        try:
            os.makedirs(os.path.join(out_folder, f))
        except OSError:
            pass
    if debug:
        log_interval = 30
    # anchor list for each scale (we have 6 scales)
    anchor_list = _build_anchor_list(cfg)
    # dataset for train and valid
    print('Loading dataset anotations...')
    (train_dataset, valid_dataset), (train_evaluation, valid_evaluation) = _build_dataset(
        cfg,
        anchor_list=anchor_list,
    )
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
        num_workers=cfg['num_workers'],
    )
    train_evaluation_loader = DataLoader(
        train_evaluation,
        batch_size=batch_size,
        collate_fn=clfn,
        num_workers=cfg['num_workers'],
    )
    valid_evaluation_loader = DataLoader(
        valid_evaluation,
        batch_size=batch_size,
        collate_fn=clfn,
        num_workers=cfg['num_workers'],
    )
    nb_classes = len(train_dataset.class_to_idx)
    bgid = train_dataset.class_to_idx['background']
    class_ids = list(set(train_dataset.class_to_idx.values()) - set([bgid]))
    class_ids = sorted(class_ids)
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
        model_class = getattr(model_module, cfg['model_name'])
        kw = cfg.get('model_config', {})
        model = model_class(
            num_anchors=list(map(len, aspect_ratios)), 
            num_classes=nb_classes,
            **kw     
        )
        model = model.cuda()
        model.transform = valid_dataset.transform 
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
        model.config = cfg
        print(model)
    if imbalance_strategy == 'class_weight':
        class_weight = torch.zeros(nb_classes)
        class_weight[0] = neg_weight
        class_weight[1:] = pos_weight
        class_weight = class_weight.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)
    #step_size = len(train_dataset) // batch_size
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
            #print(bp.size(), bt.size(), ct.size(), cp.size())
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
                nb = len(ct_pos) * background_ratio # 3x more neg than pos as in the paper
                cp_neg = cp_neg[indices[0:nb]]
                ct_neg = ct_neg[indices[0:nb]]
                l_classif = (cross_entropy(cp_pos, ct_pos, size_average=False) + cross_entropy(cp_neg, ct_neg, size_average=False)) / N
            elif imbalance_strategy == 'class_weight':
                l_classif = cross_entropy(cp, ct, weight=class_weight, size_average=False) / N
            elif imbalance_strategy == 'nothing':
                l_classif = cross_entropy(cp, ct, size_average=False) / N
            model.zero_grad()
            loss = w_loc * l_loc + w_classif * l_classif
            loss.backward()
            #scheduler.batch_step()
            _update_lr(optimizer, model.nb_updates, cfg['lr_schedule'])
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
                    # 1) for each class, filter low confidence predictions and then do NMS
                    # 2) concat all the bboxes from all classes
                    # 3) take to the topk (nms_topk)
                    pred_boxes = non_maximal_suppression_per_class(
                        pred_boxes, 
                        background_class_id=bgid,
                        iou_threshold=nms_iou_threshold,
                        score_threshold=nms_score_threshold)
                    pred_boxes = pred_boxes[0:nms_topk]
                    # get class names
                    gt_boxes = [(box, train_dataset.idx_to_class[class_id]) for box, class_id in gt_boxes]
                    pred_boxes = [(box, train_dataset.idx_to_class[class_id], score) for box, class_id, score in pred_boxes]
                    # draw boxes
                    pad = cfg['pad']
                    im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
                    im[pad:-pad, pad:-pad] = x
                    im = draw_bounding_boxes(im, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0), pad=pad)
                    im = draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
                    imsave(os.path.join(out_folder, 'train', 'sample_{:05d}.jpg'.format(i)), im)
                delta = time.time() - t0
                print('Draw box time {:.4f}s'.format(delta))
            model.nb_updates += 1
        #if debug and model.nb_updates % 50 != 0:
        #    continue
        epoch_time = time.time() - epoch_t0
        if epoch % eval_interval != 0:
            continue
        print('Evaluation') 
        t0 = time.time()
        model.eval()
        metrics = defaultdict(list)
        metrics['train_time'] = [epoch_time]
        for split_name, loader in (('train', train_evaluation_loader), ('valid', valid_evaluation_loader)):
            t0 = time.time()
            im_index = 0
            for batch, samples, in enumerate(loader):
                tt0 = time.time()
                X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred), masks = _predict(model, samples)

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
                    # for each example i  in mini-batch
                    gt_boxes = []
                    pred_boxes = []
                    x = X[i]
                    x = x.transpose((1, 2, 0))
                    x = x * np.array(std) + np.array(mean)
                    x = x.astype('float32')

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
                    for class_id in class_ids:
                        AP = average_precision(
                            pred_boxes, gt_boxes, class_id, 
                            iou_threshold=eval_iou_threshold, 
                            score_threshold=0.0,
                        )
                        if AP is not None:
                            metrics['AP_' + train_dataset.idx_to_class[class_id] + '_' + split_name].append(AP)
                            mAP.append(AP)
                    mAP = np.mean(mAP)
                    metrics['mAP_' + split_name].append(mAP)
                    # use the predicted boxes and groundtruth boxes to compute precision and recall
                    pred_boxes = non_maximal_suppression_per_class(
                        pred_boxes, 
                        background_class_id=bgid,
                        iou_threshold=nms_iou_threshold,
                        score_threshold=nms_score_threshold
                    )
                    pred_boxes = pred_boxes[0:nms_topk]
                    P = []
                    R = []
                    for class_id in class_ids:
                        t = [box for box, cl in gt_boxes if cl == class_id]
                        if len(t) == 0:
                            continue
                        p = [box for box, cl, score in pred_boxes if cl == class_id]
                        prec = precision(p, t, iou_threshold=eval_iou_threshold)
                        re = recall(p, t, iou_threshold=eval_iou_threshold)
                        metrics['precision_' + train_dataset.idx_to_class[class_id] + '_' + split_name].append(prec)
                        metrics['recall_' + train_dataset.idx_to_class[class_id] + '_' + split_name].append(re)
                        P.append(prec)
                        R.append(re)
                    metrics['precision_' + split_name].append(np.mean(P))
                    metrics['recall_' + split_name].append(np.mean(R))
                    gt_boxes = [(box, train_dataset.idx_to_class[class_id]) for box, class_id in gt_boxes]
                    pred_boxes = [(box, train_dataset.idx_to_class[class_id], score) for box, class_id, score in pred_boxes]
                    # draw boxes
                    pad = cfg['pad']
                    im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
                    im[pad:-pad, pad:-pad] = x
                    im = draw_bounding_boxes(im, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0), pad=pad)
                    im = draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
                    imsave(os.path.join(out_folder, 'eval_{}'.format(split_name), 'sample_{:05d}.jpg'.format(im_index)), im)
                    im_index += 1

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
    pred_boxes = non_maximal_suppression_per_class(
        pred_boxes, 
        iou_threshold=model.config['nms_iou_threshold'],
        background_class_id=model.class_to_idx['background'],
        score_threshold=model.config['nms_score_threshold'])
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


def find_aspect_ratios(*, config='config', nb=6):
    from sklearn.cluster import KMeans
    cfg = _read_config(config)
    anchor_list = _build_anchor_list(cfg)
    (dataset, _), _ = _build_dataset(cfg, anchor_list)
    A = []
    for i in range(len(dataset)):
        bb = dataset.boxes[i]
        for b in bb:
            (x, y, w, h), _ = b
            if h:
                A.append(w/h)
    clus = KMeans(n_clusters=nb)
    A = np.array(A).reshape((-1, 1))
    clus.fit(A)
    print(clus.cluster_centers_.flatten().tolist())


def draw_anchors(*, config='config', nb=10):
 
    cfg = _read_config(config)
    anchor_list = _build_anchor_list(cfg)
    (dataset, _), _ = _build_dataset(cfg, anchor_list)
    mean = cfg['mean']
    std = cfg['std']
    image_size = cfg['image_size']
    for i in range(nb):
        x, gt_boxes, encoding = dataset[i]
        gt_boxes = [
            ((x*image_size, y*image_size, w*image_size, h*image_size), dataset.idx_to_class[class_id]) 
            for ((x, y, w, h), class_id) in gt_boxes
        ]
        x = x.numpy()
        x = x.transpose((1, 2, 0))
        x = x * np.array(std) + np.array(mean)
        x = x.astype('float32')
        anchor_boxes = []
        for j, e in enumerate(encoding):
            B, C, M = e
            A = anchor_list[j]
            hcoords, wcoords, k_id = np.where(M)
            for h, w, k in zip(hcoords, wcoords, k_id):
                a = image_size * A[h, w, k]
                a = a.tolist()
                c = C[h, w, k]
                c = dataset.idx_to_class[c]
                anchor_boxes.append((a, c))
        pad = 100
        im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
        im[pad:-pad, pad:-pad] = x
        im = draw_bounding_boxes(im, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0), pad=pad)
        im = draw_bounding_boxes(im, anchor_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
        imsave(os.path.join(cfg['out_folder'], 'anchors', 'sample_{:05d}.jpg'.format(i)), im)


def _build_dataset(cfg, anchor_list):
    mean = cfg['mean']
    std = cfg['std']
    image_size = cfg['image_size']
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = cfg['dataset']
    if dataset == 'COCO':
        train_split = 'train' + str(cfg['dataset_version'])
        val_split = 'val' + str(cfg['dataset_version'])
        kwargs = dict(
            folder=cfg['dataset_root_folder'],
            anchor_list=anchor_list,
            iou_threshold=cfg['bbox_encoding_iou_threshold'],
            classes=cfg['classes'],
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
            iou_threshold=cfg['bbox_encoding_iou_threshold'],
            classes=cfg['classes'],
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
    return (train_dataset, valid_dataset), (train_evaluation, valid_evaluation)


def _build_anchor_list(cfg):
    scales = cfg['scales']
    aspect_ratios = cfg['aspect_ratios']
    offset = cfg['offset']
    fs = cfg['feature_map_sizes']
    assert len(scales) == len(aspect_ratios) == len(fs)
    nb = len(scales)
    anchor_list = [build_anchors(scales[i], offset=offset, feature_map_size=fs[i], aspect_ratios=aspect_ratios[i]) for i in range(nb)]
    return anchor_list


def _read_config(config):
    cfg = {}
    exec(open(config).read(), {}, cfg)
    return cfg


if __name__ == '__main__':
    run([train, test, draw_anchors, find_aspect_ratios])
