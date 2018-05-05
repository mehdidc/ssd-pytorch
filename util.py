from collections import defaultdict
import numpy as np
import torch
import cv2
from itertools import chain
from joblib import Memory
from hashlib import md5
import pandas as pd

X, Y, W, H, CLASS_ID, MASK = list(range(6))
BOUNDING_BOX = list(range(4))
eps = 1e-10

def memo(f):
    cache = {}
    def f_wrapped(*args, **kwargs):
        m = md5()
        for a in args:
            m.update(str(a)[0:100].encode('utf8'))
        h = m.hexdigest()
        if h in cache:
            return cache[h]
        else:
            cache[h] = f(*args, **kwargs)
            return cache[h]
    return f_wrapped

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

    np.array of shape (nb, 4, feature_map_size, feature_map_size) where
        - nb is len(aspect_ratios)
        - we have 4 in the second order to represent x, y, w, h of each bounding box
    """
    A = np.empty((len(aspect_ratios), 4, feature_map_size, feature_map_size))
    for i, ar in enumerate(aspect_ratios):
        for j in range(feature_map_size):
            for k in range(feature_map_size):
                x = image_size * ((k + 0.5) / feature_map_size)
                y = image_size * ((j + 0.5) / feature_map_size)
                w = image_size * (scale * np.sqrt(ar))
                h = image_size * (scale / np.sqrt(ar))
                A[i, :, j, k] = x, y, w, h
    return A


def encode_bounding_box_list_many_to_one(bbox_list, anchors, iou_threshold=0.5):
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
    
    np.array of shape (nb, 6, fh, fw) where:
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
    E = np.zeros((anchors.shape[0], 4 + 1 + 1,) + anchors.shape[2:])
    # 4 = (x, y, w, h)
    # 1 = class id
    # 1 = mask
    for k in range(anchors.shape[0]):
        for ha in range(anchors.shape[2]):
            for wa in range(anchors.shape[3]):
                if len(bbox_list) == 0:
                    continue
                ax = anchors[k, X, ha, wa]
                ay = anchors[k, Y, ha, wa]
                aw = anchors[k, W, ha, wa]
                ah = anchors[k, H, ha, wa]
                anchor = ax, ay, aw, ah
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
                E[k, MASK, ha, wa] = best_iou > iou_threshold
                E[k, X, ha, wa] = (bx - ax) / aw
                E[k, Y, ha, wa] = (by - ay) / ah
                E[k, W, ha, wa] = np.log(eps + bw / aw)
                E[k, H, ha, wa] = np.log(eps + bh / ah)
                E[k, CLASS_ID, ha, wa] = class_id
    return E


def encode_bounding_box_list_one_to_one(bbox_list, anchors, iou_threshold=0.5):
    E = np.zeros((anchors.shape[0], 4 + 1 + 1,) + anchors.shape[2:])
    # 4 = (x, y, w, h)
    # 1 = class id
    # 1 = mask
    for (bbox, class_id) in bbox_list:
        best_iou = 0.
        best_k = 0
        best_wa = 0
        best_ha = 0
        best_bbox = None
        for k in range(anchors.shape[0]):
            for ha in range(anchors.shape[2]):
                for wa in range(anchors.shape[3]):
                    ax = anchors[k, X, ha, wa]
                    ay = anchors[k, Y, ha, wa]
                    aw = anchors[k, W, ha, wa]
                    ah = anchors[k, H, ha, wa]
                    anchor = ax, ay, aw, ah
                    iou_ = iou(anchor, bbox)
                    if iou_ >= best_iou:
                        best_iou = iou_
                        best_k = k
                        best_ha = ha
                        best_wa = wa
                        best_bbox = ax, ay, aw, ah
        ax, ay, aw, ah = best_bbox
        bx, by, bw, bh = bbox
        E[best_k, MASK, best_ha, best_wa] = 1#best_iou > iou_threshold
        E[best_k, X, best_ha, best_wa] = ((bx - ax) / aw)
        E[best_k, Y, best_ha, best_wa] = ((by - ay) / ah)
        E[best_k, W, best_ha, best_wa] = np.log(eps + bw / aw)
        E[best_k, H, best_ha, best_wa] = np.log(eps + bh / ah)
        E[best_k, CLASS_ID, best_ha, best_wa] = class_id
    return E


encode_bounding_box_list = encode_bounding_box_list_many_to_one


def decode_bounding_box_list(E, anchors, include_scores=False):
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
    for k in range(anchors.shape[0]):
        for ha in range(anchors.shape[2]):
            for wa in range(anchors.shape[3]):
                bx, by, bw, bh = E[k, BOUNDING_BOX, ha, wa]
                ax, ay, aw, ah = anchors[k, BOUNDING_BOX, ha, wa]
                class_id = E[k, CLASS_ID, ha, wa]
                x = bx * aw + ax
                y = by * ah + ay
                w = np.exp(bw) * aw
                h = np.exp(bh) * ah
                bbox = x, y, w, h
                if include_scores is True:
                    scores = E[k, len(BOUNDING_BOX):, ha, wa]
                    scores = softmax(scores, axis=0)
                    class_id = scores.argmax()
                    score = scores[class_id]
                    if class_id > 0: # not background
                        bbox_list.append((bbox, class_id, score))
                else:
                    class_id = E[k, CLASS_ID, ha, wa]
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


def draw_bounding_boxes(image, bbox_list, color=[1.0, 1.0, 1.0], text_color=(1, 1, 1), font=cv2.FONT_HERSHEY_PLAIN, font_scale=1.0):
    image = image.copy()
    for bbox, class_name in bbox_list:
        bbox = uncenter_bounding_box(bbox)
        x, y, w, h = bbox
        if x < image.shape[1] and y < image.shape[0] and x + w < image.shape[1] and y + h < image.shape[0]:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color)
        text = class_name
        text_x = int(x)
        text_y = int(y)
        image = cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, 2, cv2.LINE_AA)
    return image

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


def rescale_bounding_box(bbox, from_size, to_size):
    """
    Rescale a bounding box to adapt its coordinates to an image of different size
    We use this because we rescale images from (w, h) to some some constant size like
    (300, 300)
    """
    x, y, w, h = bbox
    x = (x / from_size[0]) * to_size[0]
    y = (y / from_size[1]) * to_size[1]
    w = (w / from_size[0]) * to_size[0]
    h = (h / from_size[1]) * to_size[1]
    return x, y, w, h

def center_bounding_box(bbox):
    x, y, w, h = bbox
    x = x + w / 2
    y = y + h / 2
    return x, y, w, h

def uncenter_bounding_box(bbox):
    x, y, w, h = bbox
    x = x - w / 2
    y = y - h / 2
    return x, y, w, h

def softmax(x, axis=1):
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True) # only difference"

def smooth_l1(x, y):
    d = torch.abs(x - y)
    return (d<1).float() * 0.5 * d**2 + (d>=1).float() * (d - 0.5)


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


if __name__ == '__main__':
    from skimage.io import imread
    boxes = [  
        ((30, 30, 50, 50), 1),
        ((0, 0, 20, 20), 2)
    ]
    print(boxes)
    A = build_anchors(scale=0.2, feature_map_size=30)
    E = encode_bounding_box_list_one_to_one(boxes, A)
    boxes = decode_bounding_box_list(E, A)
    print(boxes)
