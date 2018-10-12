import time
import os
import numpy as np
from collections import defaultdict
from clize import run
from skimage.io import imsave
import pandas as pd
from sklearn.cluster import KMeans

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
from dataset import WIDER
from dataset import SubSample


import pyximport; pyximport.install()
from bounding_box import decode_bounding_box_list
from bounding_box import non_maximal_suppression_per_class
from bounding_box import non_maximal_suppression
from bounding_box import precision
from bounding_box import recall
from bounding_box import build_anchors
from bounding_box import draw_bounding_boxes
from bounding_box import average_precision
from bounding_box import get_probas
from bounding_box import binary_cross_entropy_with_logits

from optim import FocalLoss

cudnn.benchmark = True


def train(*, config='config', resume=False):
    print('Read config "{}"'.format(config))
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
    negative_per_positive = cfg['negative_per_positive']
    if imbalance_strategy == 'class_weight':
        pos_weight = cfg['pos_weight']
        neg_weight = cfg['neg_weight']
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
        n = 10
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
        use_discrete_coords = model.use_discrete_coords
        coords_discretization = model.coords_discretization
    else:
        use_discrete_coords = cfg['use_discrete_coords']
        if use_discrete_coords:
            coords_discretization = torch.linspace(
                cfg['discrete_coords_min'], 
                cfg['discrete_coords_max'],
                cfg['discrete_coords_nb']
            )
            num_coords = 4 * len(coords_discretization) # discretization values for each x,y,w,h
        else:
            num_coords = 4
            coords_discretization = None
        if 'init_from' in cfg:
            print('Init from {}'.format(cfg['init_from']))
            model = torch.load(cfg['init_from'])
        else:
            model_class = getattr(model_module, cfg['model_name'])
            kw = cfg.get('model_config', {})
            model = model_class(
                num_anchors=list(map(len, aspect_ratios)), 
                num_classes=nb_classes,
                num_coords=num_coords,
                **kw, 
            )
        model.use_discrete_coords = use_discrete_coords
        model.num_coords = num_coords
        model.coords_discretization = coords_discretization
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
    classif_loss_name = cfg.get('classif_loss', 'cross_entropy')
    model.classif_loss_name = classif_loss_name
    if classif_loss_name == 'cross_entropy':
        classif_loss = cross_entropy
    elif classif_loss_name == 'binary_cross_entropy':
        classif_loss = binary_cross_entropy_with_logits
    elif classif_loss_name == 'focal_loss':
        classif_loss = FocalLoss(
            gamma=cfg.get('focal_loss_gamma', 2),
            alpha=cfg.get('focal_loss_alpha', None),
        )
    else:
        raise ValueError('Unknown classif loss : {}'.format(classif_loss_name))

    if imbalance_strategy == 'class_weight':
        class_weight = torch.zeros(nb_classes)
        class_weight[0] = neg_weight
        class_weight[1:] = pos_weight
        class_weight = class_weight.cuda()
    optimizer_cls = getattr(torch.optim, cfg['optim_algo'])
    optimizer = optimizer_cls(model.parameters(), lr=0, **cfg['optim_params'])
    
    for epoch in range(first_epoch, num_epoch):
        epoch_t0 = time.time()
        model.train()
        for batch, samples, in enumerate(train_loader):
            t0 = time.time()
            X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred) = _predict(model, samples)
            # X is batch of image
            # Y is groundtruth output
            # Ypred is predicted output
            # bbox_true are groundtruth bounding boxes extracted from Y
            # bbox_pred are predicted bounding boxes extracted from Ypred
            # class_true are groundtruth classes extracted from Y
            # class_pred are predicted classes extracted from Ypred
            m = (class_true != bgid).view(-1)
            ind = m.nonzero().view(-1)
            bt = bbox_true
            bp = bbox_pred
            ct = class_true
            cp = class_pred
            N = max(len(ind), 1.0)
            # localization loss
            if use_discrete_coords:
                l_loc = _discrete_coords_loss(bp[ind], bt[ind], coords_discretization)
            else:
                l_loc = smooth_l1_loss(bp[ind], bt[ind], size_average=False) / N
            # classif loss
            if imbalance_strategy == 'hard_negative_mining':
                ind = torch.arange(len(ct))
                pos = ind[(ct.data.cpu() > 0)].long().cuda()
                neg = ind[(ct.data.cpu() == 0)].long().cuda()
                ct_pos = ct[pos]
                cp_pos = cp[pos]
                ct_neg = ct[neg]
                cp_neg = cp[neg]
                cp_neg_loss = classif_loss(cp_neg, ct_neg, reduce=False)
                cp_neg_loss = cp_neg_loss.cuda()
                vals, indices = cp_neg_loss.sort(descending=True)
                nb = len(ct_pos) * negative_per_positive
                cp_neg = cp_neg[indices[0:nb]]
                ct_neg = ct_neg[indices[0:nb]]
                l_classif = (classif_loss(cp_pos, ct_pos, size_average=False) + classif_loss(cp_neg, ct_neg, size_average=False)) / N
            elif imbalance_strategy == 'hard_negative_mining_with_sampling':
                ind = torch.arange(len(ct))
                pos = ind[(ct.data.cpu() > 0)].long().cuda()
                neg = ind[(ct.data.cpu() == 0)].long().cuda()
                ct_pos = ct[pos]
                cp_pos = cp[pos]
                ct_neg = ct[neg]
                cp_neg = cp[neg]
                nb = min(len(ct_pos) * negative_per_positive, len(ct_neg))
                cp_neg_loss = classif_loss(cp_neg, ct_neg, reduce=False)
                proba_sel = torch.nn.Softmax(dim=0)(cp_neg_loss)
                proba_sel = proba_sel.cuda()
                indices = torch.multinomial(proba_sel, nb)
                cp_neg = cp_neg[indices]
                ct_neg = ct_neg[indices]
                l_classif = (classif_loss(cp_pos, ct_pos, size_average=False) + classif_loss(cp_neg, ct_neg, size_average=False)) / N
            elif imbalance_strategy == 'undersampling':
                ind = torch.arange(len(ct))
                pos = ind[(ct.data.cpu() > 0)].long().cuda()
                neg = ind[(ct.data.cpu() == 0)].long().cuda()
                ct_pos = ct[pos]
                cp_pos = cp[pos]
                ct_neg = ct[neg]
                cp_neg = cp[neg]
                nb = len(ct_pos) * negative_per_positive
                inds = torch.from_numpy(np.random.randint(0, len(ct_neg), nb))
                inds = inds.long().cuda()
                ct_neg = ct_neg[inds]
                cp_neg = cp_neg[inds]
                l_classif = (classif_loss(cp_pos, ct_pos, size_average=False) + classif_loss(cp_neg, ct_neg, size_average=False)) / N
            elif imbalance_strategy == 'class_weight':
                # TODO make it work if classif_loss is "binary_cross_entropy", it does not
                # work in that case
                l_classif = classif_loss(cp, ct, weight=class_weight, size_average=False) / N
            elif imbalance_strategy == 'nothing':
                l_classif = classif_loss(cp, ct, size_average=False) / N
            else:
                raise ValueError('unknown imbalance strategy : {}'.format(imbalance_strategy))
            model.zero_grad()
            loss = w_loc * l_loc + w_classif * l_classif
            loss.backward()
            _update_lr(optimizer, model.nb_updates, cfg['lr_schedule'])
            optimizer.step()
            model.avg_loss = model.avg_loss * gamma + item(loss) * (1 - gamma)
            model.avg_loc = model.avg_loc * gamma  + item(l_loc) * (1 - gamma)
            model.avg_classif = model.avg_classif * gamma + item(l_classif) * (1 - gamma)
            delta = time.time() - t0
            print('Epoch {:05d}/{:05d} Batch {:05d}/{:05d} Loss : {:.3f} Loc : {:.3f} '
                  'Classif : {:.3f} AvgTrainLoss : {:.3f} AvgLoc : {:.3f} '
                  'AvgClassif {:.3f} Time:{:.3f}s'.format(
                      epoch + 1,
                      num_epoch,
                      batch + 1, 
                      len(train_loader), 
                      item(loss), 
                      item(l_loc),
                      item(l_classif),
                      model.avg_loss,
                      model.avg_loc,
                      model.avg_classif,
                      delta
                    ))
            train_stats['loss'].append(item(loss))
            train_stats['loc'].append(item(l_loc))
            train_stats['classif'].append(item(l_classif))
            train_stats['time'].append(delta)

            if model.nb_updates % log_interval == 0:
                # reporting part
                # -- draw training samples with their predicted and true bounding boxes
                pd.DataFrame(train_stats).to_csv(train_stats_filename, index=False)
                t0 = time.time()
                torch.save(model, model_filename)
                X = X.data.cpu().numpy()
                
                B = [[b for b, c in y] for y in Y]
                B = [(np.array(b)) for b in B]
                C = [[c for b, c in y] for y in Y]
                C = [(np.array(c)) for c in C]
                # B contains groundtruth bounding boxes for each scale
                # C contains groundtruth classes for each scale
                BP = [
                    bp.data.cpu().view(bp.size(0), -1, model.num_coords, bp.size(2), bp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
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
                        if use_discrete_coords:
                            bp = _get_coords(bp, coords_discretization)

                        A = model.anchor_list[j]
                        # get groundtruth boxes
                        gt_boxes.extend(decode_bounding_box_list(
                            bt, ct, A, 
                            include_scores=False,
                            image_size=image_size,
                            variance=cfg['variance'],
                        ))
                        # get predicted boxes
                        cp = get_probas(cp, classif_loss_name, axis=3)
                        pred_boxes.extend(decode_bounding_box_list(
                            bp, cp, A, 
                            include_scores=True,
                            image_size=image_size,
                            variance=cfg['variance'],
                        ))
                    gt_boxes = [(box, class_id) for box, class_id in gt_boxes if class_id != bgid]
                    # apply non-maximal suppression to predicted boxes
                    # 1) for each class, filter low confidence predictions and then do NMS
                    # 2) concat all the bboxes from all classes
                    # 3) take to the topk (nms_topk)
                    if cfg['use_nms']:
                        pred_boxes = non_maximal_suppression_per_class(
                            pred_boxes, 
                            background_class_id=bgid,
                            iou_threshold=nms_iou_threshold,
                            score_threshold=nms_score_threshold)
                        pred_boxes = pred_boxes[0:nms_topk]
                    else:
                        pred_boxes = [
                            (box, scores.argmax(), scores.max()) 
                            for box, scores in pred_boxes if scores.argmax() != bgid and scores.max() > nms_score_threshold
                        ]
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
        epoch_time = time.time() - epoch_t0
        if epoch % eval_interval != 0:
            continue
        if cfg.get('evaluate', True) is False:
            continue
        print('Evaluation') 
        t0 = time.time()
        model.eval()
        metrics = defaultdict(list)
        metrics['train_time'] = [epoch_time]
        for split_name, loader in (('train', train_evaluation_loader), ('valid', valid_evaluation_loader)):
            t0 = time.time()
            im_index = 0
            all_gt_boxes = defaultdict(list)
            all_pred_boxes = defaultdict(list)
            for batch, samples, in enumerate(loader):
                tt0 = time.time()
                X, (Y, Ypred), (bbox_true, bbox_pred), (class_true, class_pred) = _predict(model, samples)
                X = X.data.cpu().numpy()

                B = [[b for b, c in y] for y in Y]
                B = [torch.from_numpy(np.array(b)).float() for b in B]
                C = [[c for b, c in y] for y in Y]
                C = [torch.from_numpy(np.array(c)).long() for c in C]
                 
                # B contains groundtruth bounding boxes for each scale
                # C contains groundtruth classes for each scale

                BP = [
                    bp.data.cpu().view(bp.size(0), -1, model.num_coords, bp.size(2), bp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]
                CP = [
                    cp.data.cpu().view(cp.size(0), -1, model.nb_classes, cp.size(2), cp.size(3)).permute(0, 3, 4, 1, 2).numpy() 
                for bp, cp in Ypred]

                # BP contains predicted bounding boxes for each scale
                # CP contains predicted classes for each scale
                for i in range(len(X)):
                    gt_boxes = []
                    pred_boxes = []
                    pred_boxes_per_class = defaultdict(list)
                    # for each example i  in mini-batch
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
                        if use_discrete_coords:
                            bp = _get_coords(bp, coords_discretization)
                        A = model.anchor_list[j]
                        # get groundtruth boxes
                        gt_boxes.extend(decode_bounding_box_list(
                            bt, ct, A, 
                            include_scores=False,
                            image_size=image_size,
                            variance=cfg['variance'],
                        ))
                        # get predicted boxes
                        cp = get_probas(cp, classif_loss_name, axis=3)
                        pred_boxes.extend(decode_bounding_box_list(
                            bp, cp, A, 
                            include_scores=True,
                            image_size=image_size,
                            variance=cfg['variance']
                        ))
                    gt_boxes = [(box, class_id) for box, class_id in gt_boxes if class_id != bgid]
                    for class_id in class_ids:
                        all_gt_boxes[class_id].extend([
                            (box, im_index) 
                            for box, box_class_id in gt_boxes if class_id == box_class_id]
                        )
                    # use the predicted boxes and groundtruth boxes to compute precision and recall
                    # PER image
                    pred_boxes = non_maximal_suppression_per_class(
                        pred_boxes, 
                        background_class_id=bgid,
                        iou_threshold=nms_iou_threshold,
                        score_threshold=nms_score_threshold,
                    )
                    for box, class_id, score in pred_boxes:
                        all_pred_boxes[class_id].append((box, im_index, score))
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
                print('Eval Batch {:04d}/{:04d} on split {}, Time : {:.3f}s'.format(batch, len(loader), split_name, delta))
            # mAP
            print('Compute mAP...')
            t0_ap = time.time()
            recalls_mAP = np.linspace(0, 1, 11)
            AP_for_recall = defaultdict(list)
            AP_for_class = []
            for class_id in class_ids:
                APs = average_precision(
                    all_pred_boxes[class_id], all_gt_boxes[class_id], 
                    iou_threshold=eval_iou_threshold, 
                    recalls_mAP=recalls_mAP,
                    aggregate=False,
                )
                if APs is None:
                    continue
                AP = np.mean(APs)
                AP_for_class.append(AP)
                metrics['AP_' + train_dataset.idx_to_class[class_id] + '_' + split_name].append(AP)
                for r, ap in zip(recalls_mAP, APs):
                    m = 'AP(rec_{:.2f})_{}_{}'.format(r, train_dataset.idx_to_class[class_id], split_name)
                    metrics[m].append(ap)
                    AP_for_recall[r].append(ap)
            mAP = np.mean(AP_for_class)
            metrics['mAP_' + split_name].append(mAP)
            for r in recalls_mAP:
                mAP = np.mean(AP_for_recall[r])
                metrics['mAP(rec_{:.2f})_{}'.format(r, split_name)].append(mAP)
            print('mAP computing time : {:.3f}'.format(time.time() - t0_ap))
            delta = time.time() - t0
            metrics['eval_' + split_name + '_time'] = [delta]
            print('Eval time of {}: {:.4f}s'.format(split_name, delta))
        for k in sorted(metrics.keys()):
            v = np.mean(metrics[k])
            print('{}: {:.4}'.format(k, v))
            stats[k].append(v)
        stats['epoch'].append(epoch)
        pd.DataFrame(stats).to_csv(stats_filename, index=False)
        

def _discrete_coords_loss(bp, bt, coords_discretization, **kwargs):
    # shape of bt: (n_examples, 4)
    # shape of bp : (n_examples, 4 * len(coords_discretization))
    c = Variable(coords_discretization.view(1, 1, -1)).cuda()
    bt = bt.view(bt.size(0), 4, 1)
    _, btd = torch.abs(bt - c).min(2)
    bp  = bp.view(bp.size(0) *  4, len(coords_discretization))
    btd = btd.view(btd.size(0) * 4)
    return cross_entropy(bp, btd, **kwargs)


def _get_coords(bp, coords_discretization):
    # shape of bp : (h, w, nb_anchors, 4 * len(coords_discretization))
    bp = torch.from_numpy(bp)
    bp = bp.contiguous()
    h, w, nb_anchors = bp.size(0), bp.size(1), bp.size(2)
    bp = bp.view(h * w * nb_anchors, 4, len(coords_discretization))
    _, bpd = torch.max(bp, 2)
    bpd = bpd.view(-1)
    bpd = coords_discretization[bpd]
    bpd = bpd.view(h, w, nb_anchors, 4)
    return bpd.numpy()


def _predict(model, samples):
    X = torch.stack([x for x, _, _ in samples], 0) 
    X = X.cuda()
    X = Variable(X)
    # X has shape (nb_examples, 3, image_size, image_size)
    bbox_encodings = [be for _, _, be in samples]
    Y = list(zip(*bbox_encodings))
    
    B = [[b for b, c in y] for y in Y]
    B = [torch.from_numpy(np.array(b)).float() for b in B]
    C = [[c for b, c in y] for y in Y]
    C = [torch.from_numpy(np.array(c)).long() for c in C]
    bt = [b.view(-1, 4) for b in B]
    bt = torch.cat(bt, 0)
    # B has shape (*, 6)

    ct = [c.view(-1) for c in C]
    ct = torch.cat(ct, 0)
    
    Ypred = model(X)
    bp = [b.view(b.size(0), -1, model.num_coords, b.size(2), b.size(3)).permute(0, 3, 4, 1, 2).contiguous() for b, c in Ypred]
    cp = [c.view(c.size(0), -1, model.nb_classes, c.size(2), c.size(3)).permute(0, 3, 4, 1, 2).contiguous() for b, c in Ypred]    
    
    bp = [b.view(-1, model.num_coords) for b in bp]
    bp = torch.cat(bp, 0)

    cp = [c.view(-1, model.nb_classes) for c in cp]
    cp = torch.cat(cp, 0)

    ct = Variable(ct).cuda()
    bt = Variable(bt).cuda()
    return X, (Y, Ypred), (bt, bp), (ct, cp)


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


def test(
    *,
    in_folder='test_images',
    out_folder='test_results',
    model='out/model.th', 
    score_threshold=None, 
    topk=10, 
    iou_threshold=None,
    background_threshold=0.5,
    use_nms=True, 
    out=None, 
    cuda=False):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    model = torch.load(model, map_location=lambda storage, loc: storage)
    if cuda:
        model = model.cuda()
    model.eval()
    filenames = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
    for filename in filenames:
        im = default_loader(filename)
        x = model.transform(im)
        X = x.view(1, x.size(0), x.size(1), x.size(2))
        if cuda:
            X = X.cuda()
        X = Variable(X)
        Ypred = model(X)
        BP = [
            bp.data.cpu().
            view(bp.size(0), -1, model.num_coords, bp.size(2), bp.size(3)).
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
            if model.use_discrete_coords:
                bp = _get_coords(bp, model.coords_discretization)
            A = model.anchor_list[j]
            cp = get_probas(cp, model.classif_loss_name, axis=3)
            boxes = decode_bounding_box_list(
                bp, cp, A, 
                image_size=model.image_size,
                include_scores=True,
                variance=model.config['variance']
            )
            pred_boxes.extend(boxes)
        if score_threshold is None:
            score_threshold = model.config['nms_score_threshold']
        else:
            score_threshold = float(score_threshold)
        if iou_threshold is None:
            iou_threshold = model.config['nms_iou_threshold']
        else:
            iou_threshold = float(iou_threshold)
        bgid = model.class_to_idx['background']
        idx_to_class = {i: c for c, i in model.class_to_idx.items()}
        
        if use_nms:
            pred_boxes = non_maximal_suppression_per_class(
                pred_boxes, 
                iou_threshold=iou_threshold,
                background_class_id=bgid,
                score_threshold=score_threshold
            )
        else:
            def get_box(box, scores):
                bg_score = scores[0]
                if bg_score >= background_threshold:
                    return box, 0, bg_score
                else:
                    cl= 1 + scores[1:].argmax()
                    score = scores[1:].max()
                    return box, cl, score
            pred_boxes = [get_box(box, scores) for box, scores in pred_boxes]
        pred_boxes = sorted(pred_boxes, key=lambda p:p[2], reverse=True)
        pred_boxes = [(box, class_id, score) for box, class_id, score in pred_boxes if class_id != bgid]
        pred_boxes = [(box, idx_to_class[class_id], score) for box, class_id, score in pred_boxes]
        pred_boxes = pred_boxes[0:topk]
        for box, class_name, score in pred_boxes:
            print(class_name, score)
        pad = model.config['pad']
        im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
        im[pad:-pad, pad:-pad] = x
        im = draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
        outf = os.path.join(out_folder, os.path.basename(filename))
        print(outf)
        imsave(outf, im)


def find_aspect_ratios(*, config='config', nb=6):
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


def draw_anchors(*, config='config', nb=100, only_groundtruth=False):
 
    cfg = _read_config(config)
    anchor_list = _build_anchor_list(cfg)
    (dataset, _), _ = _build_dataset(cfg, anchor_list)
    bgid = dataset.class_to_idx['background']

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
        if not only_groundtruth:
            for j, e in enumerate(encoding):
                B, C = e
                A = anchor_list[j]
                hcoords, wcoords, k_id = np.where(C != bgid)
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
        train_annotations = cfg['dataset_train_annotations'] 
        val_annotations = cfg['dataset_valid_annotations']
        train_folder = cfg['dataset_train_images_folder']
        val_folder = cfg['dataset_valid_images_folder']
        kwargs = dict(
            anchor_list=anchor_list,
            iou_threshold=cfg['bbox_encoding_iou_threshold'],
            classes=cfg['classes'],
            transform=train_transform,
            variance=cfg['variance']
        )
        train_dataset = COCO(
            annotations=train_annotations,
            images_folder=train_folder,
            data_augmentation_params=cfg['data_augmentation_params'],
            **kwargs
        )
        valid_dataset = COCO(
            annotations=val_annotations,
            images_folder=val_folder,
            data_augmentation_params={},
            **kwargs,
        )
        train_evaluation = SubSample(COCO(
            annotations=train_annotations,
            images_folder=train_folder,
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
            transform=train_transform,
            variance=cfg['variance'],
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
    elif dataset == 'WIDER':
        kwargs = dict(
            folder=cfg['dataset_root_folder'],
            anchor_list=anchor_list, 
            iou_threshold=cfg['bbox_encoding_iou_threshold'],
            transform=train_transform,
            variance=cfg['variance'],
        )
        train_dataset = WIDER(
            split='train',
            data_augmentation_params=cfg['data_augmentation_params'],
            **kwargs,
        )
        valid_dataset = WIDER(
            split='val',
            data_augmentation_params={},
            **kwargs,
        )
        train_evaluation = SubSample(WIDER(
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

def sample_hypers_and_train(config_template):
    content = open(config_template).read()
    name, config = _generate_config_from_template(content)
    config_file = 'configs/{}'.format(name)
    with open(config_file, 'w') as fd:
        fd.write(config)
    train(config=config_file)


def _generate_config_from_template(content):
    from jinja2 import Template
    import uuid
    rng = np.random.RandomState()
    name = str(uuid.uuid4())
    out_folder = '"results/{}"'.format(name)
    tpl = Template(content)

    algo = rng.choice(('"Adam"', '"SGD"'))
    if algo == '"SGD"':
        algo_params = {'momentum': 0.9, 'weight_decay': rng.choice((0, 1e-4, 3e-4))}
    elif algo == '"Adam"':
        algo_params = {}
    algo_params = str(algo_params)
    model_name = rng.choice(('"SSD_VGG"', '"SSD_Resnet"'))
    batch_size = 8
    use_discrete_coords = rng.choice((True, False))
    imbalance_strategy = rng.choice((
        'hard_negative_mining_with_sampling',
        'hard_negative_mining',
        'nothing',
        'undersampling',
    ))
    imbalance_strategy = '"{}"'.format(imbalance_strategy)
    if model_name == '"SSD_Resnet"':
        arch = rng.choice(('resnet18', 'resnet34', 'resnet50'))
        model_config = {'arch': arch}
        model_config = str(model_config)
        print(model_config)
        batch_size = 1
    else:
        model_config = {}
    classif_loss = rng.choice(('"cross_entropy"', '"binary_cross_entropy"'))
    params = {
        'w_loc': rng.choice((1, 2, 0.5, 0.1)),
        'w_classif': 1,
        'use_discrete_coords': use_discrete_coords,
        'negative_per_positive': rng.choice((1, 2, 3, 5, 10)),
        'pos_weight': 1,
        'neg_weight': 0.1,
        'out_folder': out_folder,
        'lr_init': rng.choice((0.1, 0.01, 0.001, 0.0001)),
        'optim_algo': algo,
        'optim_params': algo_params,
        'batch_size': batch_size,
        'model_name': model_name,
        'model_config': model_config,
        'imbalance_strategy': imbalance_strategy,
        'classif_loss': classif_loss,
        'patch_proba': rng.choice((0, 0.1, 0.3, 0.5)),
        'flip_proba': rng.choice((0, 0.1, 0.3, 0.5)),
    }
    res = tpl.render(**params)
    return name, res


def leaderboard(folder='results'):
    from glob import glob
    filenames = glob(os.path.join(folder, '**', 'stats.csv'))
    rows = []
    for filename in filenames:
        name = os.path.basename(os.path.dirname(filename))
        df = pd.read_csv(filename)
        d = df.iloc[-1].to_dict()
        d['name'] = name[0:10]
        try:
            cfg = _read_config(os.path.join('configs', name))
            d['model'] = cfg['model_name']
        except Exception:
            d['model'] = ''
        rows.append(d)
    df = pd.DataFrame(rows)
    df = df[[
        'name',
        'epoch',
        'model',
        'mAP_train',
        'mAP_valid',
        #'mAP(rec_0.90)_train',
        #'mAP(rec_0.90)_valid',
        'precision_train',
        'precision_valid',
        'recall_train',
        'recall_valid',
    ]]
    df = df.round(decimals=2)
    pd.options.display.width = 200
    df = df.sort_values(by='mAP_valid', ascending=False)
    print(df)


def sample_hypers_and_train(config_template):
    content = open(config_template).read()
    folder = os.path.basename(config_template)
    name, config = _generate_config_from_template(folder, content)
    config_file = 'configs/{}/{}'.format(folder, name)
    with open(config_file, 'w') as fd:
        fd.write(config)
    train(config=config_file)


def _generate_config_from_template(root_folder_name, content):
    from jinja2 import Template
    import uuid
    rng = np.random.RandomState()
    name = str(uuid.uuid4())
    out_folder = '"results/{}/{}"'.format(root_folder_name, name)
    tpl = Template(content)

    algo = rng.choice(('"Adam"', '"SGD"'))
    if algo == '"SGD"':
        algo_params = {'momentum': 0.9, 'weight_decay': rng.choice((0, 1e-4, 3e-4))}
    elif algo == '"Adam"':
        algo_params = {}
    algo_params = str(algo_params)
    model_name = rng.choice(('"SSD_VGG"', '"SSD_Resnet"'))
    batch_size = 16
    use_discrete_coords = rng.choice((True, False))
    imbalance_strategy = rng.choice((
        'hard_negative_mining_with_sampling',
        'hard_negative_mining',
        'nothing',
        'undersampling',
        'none',
    ))
    imbalance_strategy = '"{}"'.format(imbalance_strategy)
    if model_name == '"SSD_Resnet"':
        arch = rng.choice(('resnet18', 'resnet34', 'resnet50'))
        model_config = {'arch': arch}
        model_config = str(model_config)
        print(model_config)
        batch_size = 1
    else:
        model_config = {}
    classif_loss = rng.choice((
        '"cross_entropy"', 
        '"binary_cross_entropy"', 
        '"focal_loss"'
    ))
    params = {
        'w_loc': rng.choice((1, 2, 0.5, 0.1)),
        'w_classif': 1,
        'use_discrete_coords': use_discrete_coords,
        'negative_per_positive': rng.choice((1, 2, 3, 5, 10)),
        'pos_weight': 1,
        'neg_weight': 0.1,
        'out_folder': out_folder,
        'lr_init': rng.choice((0.1, 0.01, 0.001, 0.0001)),
        'optim_algo': algo,
        'optim_params': algo_params,
        'batch_size': batch_size,
        'model_name': model_name,
        'model_config': model_config,
        'imbalance_strategy': imbalance_strategy,
        'classif_loss': classif_loss,
        'patch_proba': rng.choice((0, 0.1, 0.3, 0.5)),
        'flip_proba': rng.choice((0, 0.1, 0.3, 0.5)),
    }
    res = tpl.render(**params)
    return name, res




def item(x):
    if hasattr(x, 'item'):
        return x.item()
    else:
        return float(x.data[0])


if __name__ == '__main__':
    run([
        train, 
        test, 
        draw_anchors, 
        find_aspect_ratios, 
        leaderboard, 
        sample_hypers_and_train
    ])
