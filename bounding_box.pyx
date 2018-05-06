import time
from collections import defaultdict
import numpy as np
import torch
import cv2
from itertools import chain
from joblib import Memory
from hashlib import md5
import pandas as pd

from util import softmax

cimport numpy as np


BOUNDING_BOX = X, Y, W, H = list(range(4))
eps = 1e-10

def build_anchors(image_size=300, scale=1, feature_map_size=4, aspect_ratios=(1, 2, 3, 1 / 2, 1 / 3)):
    """
    Builds a 4-order tensor that represents a set of anchors

    Parameters
    ----------

    image_size : int
        image size in pixels

    scale : float between 0 and 1
        scale of the bounding boxes relative to `image_size`

    feature_map_size : int
        size of the feature map (assumes width and height of feature maps are equal)

    aspect_ratios : list of float
        aspect ratios of bounding boxes to consider

    Returns
    -------

    np.array of shape (feature_map_size, feature_map_size, nb, 4) where
        - nb is len(aspect_ratios)
        - we have 4 in the second order to represent x, y, w, h of each bounding box
    """
    A = np.empty((feature_map_size, feature_map_size, len(aspect_ratios), 4))
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            for k, ar in enumerate(aspect_ratios):
                x = image_size * ((k + 0.5) / feature_map_size)
                y = image_size * ((j + 0.5) / feature_map_size)
                w = image_size * (scale * np.sqrt(ar))
                h = image_size * (scale / np.sqrt(ar))
                A[i, j, k] = x, y, w, h
    return A


cpdef tuple encode_bounding_box_list_many_to_one(bbox_list, np.ndarray anchors, iou_threshold=0.5):
    """
    Convert a list of groundtruth bounding boxes into into a 4-order tensor
    that the neural net have to predict.

    Parameters
    ----------

    - bbox_list : list of couples (bbox, class_id) representing bounding boxes with their
      corresponding label, where `bbox` is a 4-tuple (x, y, w, h) 
    - anchors is a 4-order tensor (nb, 4, fh, fw) where:
        - `nb` is the number of anchors per position in the feature map of size(fh, fw)
         - we have 4 numbers because each anchor is represented by its bounding box (x, y, w, h)
    - iou_threshold : float
        iou threshold which determines whether two boxes overlap or not

    Returns
    -------
    
    np.array of shape (nb, fh, fw, 6) where:
        - `nb` is the number of anchors per position in the feature map
        - `fh` and `fw` are respectively the feature map height and width
        - each position in the feature map is responsible for predicting `nb` anchors
        - we have 6 numbers per anchor because we need to predict for each bounding box:
            - x, y, w, h, class_id, mask
            - x, y, w, h represent the bounding box rectangle relative to the anchor
            - class_id is the id of the class that is contained the bounding box
            - mask determines whether a given anchor is responsible for predicting a class or not. 
              anchors that are not overlapping enough (depending on iou_threshold) with any groundtruth 
              box have mask = False, while the others have mask=True
     
    """
    B = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2], 4)).astype('float32')
    C = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2])).astype('int32')
    M = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2])).astype('bool')
    if len(bbox_list) == 0:
        return B, C, M
    cdef int nbh = anchors.shape[0]
    cdef int nbw = anchors.shape[1]
    cdef int nbk = anchors.shape[2]
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
                bx, by, bw, bh = best_bbox
                B[ha, wa, k, X] = (bx - ax) / aw
                B[ha, wa, k, Y] = (by - ay) / ah
                B[ha, wa, k, W] = np.log(eps + bw / aw)
                B[ha, wa, k, H] = np.log(eps + bh / ah)
                C[ha, wa, k] = class_id
                M[ha, wa, k] = best_iou > iou_threshold
    return B, C, M


cpdef tuple encode_bounding_box_list_one_to_one(bbox_list, np.ndarray anchors, iou_threshold=0.5):
    B = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2], 4)).astype('float32')
    C = np.zeros((anchors.shape[0], anchors.shape[1], anchors.shape[2])).astype('int32')
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
        
        B[best_ha, best_wa, best_k, X] = ((bx - ax) / aw)
        B[best_ha, best_wa, best_k, Y] = ((by - ay) / ah)
        B[best_ha, best_wa, best_k, W] = np.log(eps + bw / aw)
        B[best_ha, best_wa, best_k, H] = np.log(eps + bh / ah)
        C[best_ha, best_wa, best_k] = class_id
        M[best_ha, best_wa, best_k] = True#best_iou > iou_threshold
    return B, C, M


def decode_bounding_box_list(B, C, anchors, include_scores=False):
    """
    Convert a 4-th order tensor of encoded bounding boxes into a list of
    detected bounding boxes
    
    Parameters
    ----------

    E : n.array of shape (nb, 5, fh, fw)
        - nb is the number of anchors per feature map position
        - we have 5 numbers because each bounding box (4) + class_id
    """
    bbox_list = []
    for ha in range(anchors.shape[0]):
        for wa in range(anchors.shape[1]):
            for k in range(anchors.shape[2]):
                bx, by, bw, bh = B[ha, wa, k]
                ax, ay, aw, ah = anchors[ha, wa, k]
                x = bx * aw + ax
                y = by * ah + ay
                w = np.exp(bw) * aw
                h = np.exp(bh) * ah
                bbox = x, y, w, h
                if include_scores is True:
                    scores = C[ha, wa, k]
                    scores = softmax(scores, axis=0)
                    class_id = scores.argmax()
                    score = scores[class_id]
                    if class_id > 0: # not background
                        bbox_list.append((bbox, class_id, score))
                else:
                    class_id = C[ha, wa, k]
                    if class_id > 0: # not background
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
    x = x - w / 2
    y = y - h / 2
    return x, y, w, h

def non_maximal_suppression_per_class(bbox_list, iou_threshold=0.5):
    classes = defaultdict(list)
    for bbox, class_id, score in bbox_list:
        classes[class_id].append((bbox, class_id, score))
    bblist_all = []
    for cl, bblist in classes.items():
        bblist = non_maximal_suppression(bblist, iou_threshold=iou_threshold)
        bblist_all.extend(bblist)
    return bblist_all


def non_maximal_suppression(bbox_list, iou_threshold=0.5):
    """
    
    Parameters
    ----------
    
    bbox_list : list of 3-tuples (bbox, class_id, score)
        where bbox is a 4-tuple (x, y, w, h),
        class_id is the class with the max score
        score is the non-background (objectness) score
    iou_threshold : float

    Returns
    -------

    list of 2-tuples (bbox, class_id)
    """
    L = set(bbox_list)
    score = lambda b:b[2]
    final_bbox_list = []
    f = []
    while len(L) > 0:
        cur = max(L, key=score)
        bbox, class_id, _ = cur
        remove = [cur]
        for l in L:
            bbox_, _, _ = l
            if iou(bbox, bbox_) > iou_threshold:
                remove.append(l)
        remove = set(remove)
        for r in remove:
            L.remove(r)
        final_bbox_list.append((bbox, class_id))
    return final_bbox_list

def precision(bbox_pred_list, bbox_true_list, iou_threshold=0.5):
    if len(bbox_pred_list) == 0 or len(bbox_true_list) == 0:
        return 0
    M = matching_matrix(bbox_pred_list, bbox_true_list, iou_threshold=iou_threshold)
    return (M.sum(axis=0) > 0).astype('float32').mean()


def recall(bbox_pred_list, bbox_true_list, iou_threshold=0.5):
    if len(bbox_pred_list) == 0 or len(bbox_true_list) == 0:
        return 0
    M = matching_matrix(bbox_pred_list, bbox_true_list, iou_threshold=iou_threshold)
    return (M.sum(axis=1) > 0).astype('float32').mean()


def matching_matrix(bbox_pred_list, bbox_true_list, iou_threshold=0.5):
    M = np.zeros((len(bbox_pred_list), len(bbox_true_list)))
    for i, (bp, cp) in enumerate(bbox_pred_list):
        for j, (bt, ct) in enumerate(bbox_true_list):
            if iou(bt, bp) > iou_threshold and cp == ct:
                M[i, j] = 1
    return M
