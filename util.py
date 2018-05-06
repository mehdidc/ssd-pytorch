import time
from collections import defaultdict
import numpy as np
import torch
import cv2
from itertools import chain
from joblib import Memory
from hashlib import md5
import pandas as pd



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
