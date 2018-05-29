import time
from collections import defaultdict
import numpy as np
import torch
import cv2
from itertools import chain
from hashlib import md5
import pandas as pd

cimport numpy as np

X, Y, W, H = 0, 1, 2, 3
eps = 1e-10

def build_anchors(scale=1, feature_map_size=4, offset=0.5, aspect_ratios=(1, 2, 3, 1. / 2, 1. / 3)):
    A = np.empty((feature_map_size, feature_map_size, len(aspect_ratios), 4))
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            for k, ar in enumerate(aspect_ratios):
                x = (k + offset) / feature_map_size
                y = (j + offset) / feature_map_size
                w = (scale * np.sqrt(ar))
                h = (scale / np.sqrt(ar))
                A[i, j, k] = x, y, w, h
    return A


cpdef list encode_bounding_box_list_many_to_one(
        bbox_list, 
        anchors_list, 
        variance=[0.1, 0.1, 0.2, 0.2], 
        background_class_id=0, 
        iou_threshold=0.5):

    E = []
    for anchors in anchors_list:
        B = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2], 4)).astype('float32')
        C = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2])).astype('int32')
        C[:] = background_class_id
        M = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2])).astype('bool')
        E.append((B, C, M)) 
    if len(bbox_list) == 0:
        return E
    # For each groundtruth box, match the best anchor
    for j, (bbox, class_id) in enumerate(bbox_list):
        best_iou = 0.
        best_bbox = None
        best_k = 0
        best_ha = 0
        best_wa = 0
        best_scale = 0
        for i, anchors in enumerate(anchors_list):
            nbh = anchors.shape[0]
            nbw = anchors.shape[1]
            nbk = anchors.shape[2]
            for ha in range(nbh):
                for wa in range(nbw):
                    for k in range(nbk):
                        anchor = ax, ay, aw, ah = anchors[ha, wa, k]
                        iou_ = iou(anchor, bbox)
                        if iou_ < best_iou:
                            continue
                        best_iou = iou_
                        best_bbox = anchor
                        best_k = k
                        best_ha = ha
                        best_wa = wa
                        best_scale = i
        ax, ay, aw, ah = best_bbox
        bx, by, bw, bh = bbox
        B, C, M = E[best_scale]
        B[best_ha, best_wa, best_k, X] = ((bx - ax) / aw) / variance[0]
        B[best_ha, best_wa, best_k, Y] = ((by - ay) / ah) / variance[1]
        B[best_ha, best_wa, best_k, W] = (np.log(eps + bw / aw)) / variance[2]
        B[best_ha, best_wa, best_k, H] = (np.log(eps + bh / ah)) / variance[3]
        C[best_ha, best_wa, best_k] = class_id 
        M[best_ha, best_wa, best_k] = True
    # match each anchor to the groundtruth box with best iou
    # if the best iou > iou_threshold
    for i, anchors in enumerate(anchors_list):
        B, C, M = E[i]
        for ha in range(nbh):
            for wa in range(nbw):
                for k in range(nbk):
                    anchor = ax, ay, aw, ah = anchors[ha, wa, k]
                    best_iou = 0.
                    best_bbox = bbox_list[0]
                    # Find the groundtruth box that best matches
                    # the anchor box
                    for j, (bbox, class_id) in enumerate(bbox_list):
                        iou_ = iou(anchor, bbox)
                        if iou_ < best_iou:
                            continue
                        best_iou = iou_
                        best_bbox = bbox
                    if M[ha, wa, k] == False:
                        bx, by, bw, bh = best_bbox
                        B[ha, wa, k, X] = ((bx - ax) / aw) / variance[0]
                        B[ha, wa, k, Y] = ((by - ay) / ah) / variance[1]
                        B[ha, wa, k, W] = (np.log(eps + bw / aw)) / variance[2]
                        B[ha, wa, k, H] = (np.log(eps + bh / ah)) / variance[3]
                        C[ha, wa, k] = class_id if best_iou >= iou_threshold else background_class_id
                        M[ha, wa, k] = best_iou >= iou_threshold
    return E


cpdef tuple encode_bounding_box_list_one_to_one(
    bbox_list, 
    np.ndarray anchors, 
    variance=[0.1, 0.1, 0.2, 0.2], 
    background_class_id=0, 
    iou_threshold=0.5):

    B = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2], 4)).astype('float32')
    C = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2])).astype('int32')
    C[:] = background_class_id
    M = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2])).astype('bool')
    cdef int nbh = anchors.shape[0]
    cdef int nbw = anchors.shape[1]
    cdef int nbk = anchors.shape[2]
    cdef float best_iou
    cdef int best_k
    cdef int best_wa
    cdef int best_ha
    cdef float iou_
    for (bbox, class_id) in bbox_list:
        best_iou = 0.
        best_k = 0
        best_wa = 0
        best_ha = 0
        best_bbox = None
        for ha in range(nbh):
            for wa in range(nbw):
                for k in range(nbk):
                    anchor = ax, ay, aw, ah = anchors[ha, wa, k]
                    iou_ = iou(anchor, bbox)
                    if iou_ >= best_iou:
                        best_iou = iou_
                        best_k = k
                        best_ha = ha
                        best_wa = wa
                        best_bbox = ax, ay, aw, ah
        ax, ay, aw, ah = best_bbox
        bx, by, bw, bh = bbox
        
        B[best_ha, best_wa, best_k, X] = (((bx - ax) / aw)) / variance[0]
        B[best_ha, best_wa, best_k, Y] = (((by - ay) / ah)) / variance[1]
        B[best_ha, best_wa, best_k, W] = (np.log(eps + bw / aw)) / variance[2]
        B[best_ha, best_wa, best_k, H] = (np.log(eps + bh / ah)) / variance[3]
        C[best_ha, best_wa, best_k] = class_id
        M[best_ha, best_wa, best_k] = True
    return B, C, M


def decode_bounding_box_list(B, C, anchors, image_size=300 ,variance=[0.1, 0.1, 0.2, 0.2], include_scores=False):
    bbox_list = []
    for ha in range(anchors.shape[0]):
        for wa in range(anchors.shape[1]):
            for k in range(anchors.shape[2]):
                bx, by, bw, bh = B[ha, wa, k]
                bx = bx * variance[0]
                by = by * variance[1]
                bw = bw * variance[2]
                bh = bh * variance[3]
                ax, ay, aw, ah = anchors[ha, wa, k]
                x = bx * aw + ax
                y = by * ah + ay
                w = np.exp(bw) * aw
                h = np.exp(bh) * ah
                bbox = x, y, w, h
                bbox = unnormalize_bounding_box(bbox, (image_size, image_size))
                if include_scores is True:
                    scores = C[ha, wa, k]
                    scores = softmax(scores, axis=0)
                    class_id = np.argmax(scores)
                    bbox_list.append((bbox, scores))
                else:
                    class_id = C[ha, wa, k]
                    bbox_list.append((bbox, class_id))
    return bbox_list 


def iou(bbox1, bbox2):
    """
    Computes the intersection over union (IOU) between two boxes

    Parameters
    ----------

    bbox1 : 4-tuple (x, y, w, h)
    bbox2 : 4-tuple (x, y, w, h)

    Returns
    -------

    float
    """
    cdef float x, y, w, h, xx, yy, ww, hh
    cdef float winter, hinter
    bbox1 = uncenter_bounding_box(bbox1)
    bbox2 = uncenter_bounding_box(bbox2)
    x, y, w, h = bbox1
    xx, yy, ww, hh = bbox2
    winter = min(x + w, xx + ww) - max(x, xx)
    hinter = min(y + h, yy + hh) - max(y, yy)
    if winter < 0 or hinter < 0:
        inter = 0
    else:
        inter = winter * hinter
    union = w * h + ww * hh - inter
    return inter / union


def uncenter_bounding_box(bbox):
    cdef float x, y, w, h
    x, y, w, h = bbox
    return x - w / 2, y - h / 2, w, h


def rescale_bounding_box(bbox, from_size, to_size):
    cdef float x, y, w, h
    x, y, w, h = bbox
    x = (x / from_size[0]) * to_size[0]
    y = (y / from_size[1]) * to_size[1]
    w = (w / from_size[0]) * to_size[0]
    h = (h / from_size[1]) * to_size[1]
    return x, y, w, h


def center_bounding_box(bbox):
    cdef float x, y, w, h
    x, y, w, h = bbox
    x = x + w / 2
    y = y + h / 2
    return x, y, w, h


def uncenter_bounding_box(bbox):
    cdef float x, y, w, h
    x, y, w, h = bbox
    x = x - w / 2
    y = y - h / 2
    return x, y, w, h


def normalize_bounding_box(bbox, size):
    cdef float x, y, w, h
    x, y, w, h = bbox
    x /= size[0]
    w /= size[0]
    y /= size[1]
    h /= size[1]
    return x, y, w, h


def unnormalize_bounding_box(bbox, size):
    cdef float x, y, w, h
    x, y, w, h = bbox
    x *= size[0]
    w *= size[0]
    y *= size[1]
    h *= size[1]
    return x, y, w, h


def softmax(x, axis=1):
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True) # only difference"


def non_maximal_suppression_per_class(bbox_list, background_class_id=0, iou_threshold=0.5, score_threshold=0.0):
    bb_scores = bbox_list[0]
    bb, scores = bb_scores
    nb_classes = len(scores)
    bblist_all = []
    for cl in range(nb_classes):
        if cl == background_class_id:
            continue
        bb = [(box, scores[cl]) for box, scores in bbox_list if scores[cl] >= score_threshold]
        bb = non_maximal_suppression(bb, iou_threshold=iou_threshold)
        bb = [(b, cl, score) for (b, score) in bb]
        bblist_all.extend(bb)
    
    def score_fn(b):
        bbox, cl, score = b
        return score
    bblist_all = sorted(bblist_all, key=score_fn, reverse=True)
    #bblist_all = [(box, cl) for (box, cl, score) in bblist_all]
    return bblist_all


def non_maximal_suppression(list bbox_list, float iou_threshold=0.5):
    def score_fn(k):
        bbox, score = k
        return score
    L = sorted(bbox_list, key=score_fn, reverse=True)
    final_bbox_list = []
    f = []
    while len(L) > 0:
        cur = L[0] 
        bbox, score = cur
        remove = [cur]
        for l in L:
            bbox_, _ = l
            if iou(bbox, bbox_) >= iou_threshold:
                remove.append(l)
        remove = set(remove)
        L = [l for l in L if l not in remove]
        final_bbox_list.append((bbox, score)) 
    return final_bbox_list


def precision(bbox_pred_list, bbox_true_list, iou_threshold=0.5):
    if len(bbox_pred_list) == 0 or len(bbox_true_list) == 0:
        return 0
    M = matching_matrix(bbox_pred_list, bbox_true_list, iou_threshold=iou_threshold)
    return (M.sum(axis=1) > 0).astype('float32').mean()


def recall(bbox_pred_list, bbox_true_list, iou_threshold=0.5):
    if len(bbox_pred_list) == 0 or len(bbox_true_list) == 0:
        return 0
    M = matching_matrix(bbox_pred_list, bbox_true_list, iou_threshold=iou_threshold)
    return (M.sum(axis=0) > 0).astype('float32').mean()

def average_precision(
    bbox_pred_list_with_scores, bbox_true_list, 
    class_id,
    recalls_mAP=np.linspace(0, 1, 11),
    iou_threshold=0.5,
    score_threshold=0.0):
    
    # Check 
    #https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    # for more info

    # -- get prediction list of bboxes 
    def score_fn(k):
        bbox, scores = k
        score = scores[class_id]
        return score
    bbox_pred_list_with_scores = sorted(bbox_pred_list_with_scores, key=score_fn, reverse=True)
    bbox_pred_list = [bbox for bbox, scores in bbox_pred_list_with_scores if scores[class_id] >= score_threshold]
    
    # --- get true list of bboxes
    bbox_true_list = [bbox for (bbox, cl) in bbox_true_list if cl == class_id]
    if len(bbox_true_list) == 0:
        return None
    
    # --- compute precison and recall for each rank
    cdef np.ndarray R = np.zeros(len(bbox_true_list))
    cdef int nb_precision = 0
    cdef int nb_recall = 0
    precisions = []
    recalls = []
    for i, bbox_pred in enumerate(bbox_pred_list):
        for j, bbox_true in enumerate(bbox_true_list):
            if iou(bbox_pred, bbox_true) >= iou_threshold:
                if R[j] == 0:
                    R[j] = 1
                    nb_recall += 1
                nb_precision += 1
        p = nb_precision / (i + 1)
        r = nb_recall / len(bbox_true_list)
        precisions.append(p)
        recalls.append(r)
    # --- compute mAP
    vals = []
    for r in recalls_mAP:
       val = max((precisions[i] for i in range(len(precisions)) if recalls[i] >= r), default=0)
       vals.append(val)
    return np.mean(vals)


def matching_matrix(bbox_pred_list, bbox_true_list, iou_threshold=0.5):
    M = np.zeros((len(bbox_pred_list), len(bbox_true_list)))
    for i, bp in enumerate(bbox_pred_list):
        for j, bt in enumerate(bbox_true_list):
            if iou(bt, bp) >= iou_threshold:
                M[i, j] = 1
    return M


def box_in_box(box_small, box_big):
    cdef float bsx, bsy, bsw, bsh
    cdef float bbx, bby, bbw, bbh
    bsx, bsy, bsw, bsh = box_small
    bbx, bby, bbw, bbh = box_big
    if not (bsx >= bbx and bsx + bsw <= bbx + bbw):
        return False
    if not (bsy >= bby and bsy + bsh <= bby + bbh):
        return False
    return True


def draw_bounding_boxes(
    image,
    bbox_list, 
    color=[1.0, 1.0, 1.0], 
    text_color=(1, 1, 1), 
    font=cv2.FONT_HERSHEY_PLAIN, 
    font_scale=1.0,
    pad=0):
    for bb in bbox_list:
        if len(bb) == 3:
            bbox, class_name, score = bb
        elif len(bb) == 2:
            bbox, class_name = bb
            score = None
        else:
            raise ValueError(bb)
        bbox = uncenter_bounding_box(bbox)
        x, y, w, h = bbox
        x = int(x) + pad
        y = int(y) + pad
        w = int(w)
        h = int(h)
        if x > image.shape[1]:
            continue
        if x + w > image.shape[1]:
            continue
        if y > image.shape[0]:
            continue
        if y + h > image.shape[1]:
            continue
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color)
        if score:
            text = '{}({:.2f})'.format(class_name, score)
        else:
            text = class_name
        image = cv2.putText(image, text, (x, y), font, font_scale, text_color, 2, cv2.LINE_AA)
    return image
