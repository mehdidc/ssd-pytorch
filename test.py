import numpy as np
import pyximport; pyximport.install()
from bounding_box import encode_bounding_box_list_many_to_one
from bounding_box import encode_bounding_box_list_one_to_one
from bounding_box import decode_bounding_box_list
from util import build_anchors

def random_box():
    x, y, w, h = np.random.randint(0, 300, size=4)
    return x, y, w, h

def main():
    boxes = [  
        (random_box(), 1)
        for _ in range(1)
    ]
    print(boxes)
    A = build_anchors(scale=0.2, feature_map_size=30)
    B, C, M = encode_bounding_box_list_one_to_one(boxes, A)
    D = decode_bounding_box_list(B, C, A)
    for bbox in D:
        print(bbox)

if __name__ == '__main__':
    main()
