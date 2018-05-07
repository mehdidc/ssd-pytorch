import numpy as np
import pyximport; pyximport.install()
from bounding_box import encode_bounding_box_list_many_to_one
from bounding_box import encode_bounding_box_list_one_to_one
from bounding_box import decode_bounding_box_list
from bounding_box import build_anchors

from itertools import chain

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
    B = build_anchors(scale=0.5, feature_map_size=20)
    (Ba, Ca, Ma), (Bb, Cb, Mb) = encode_bounding_box_list_many_to_one(boxes, [A, B])
    
    Da = decode_bounding_box_list(Ba, Ca, A)
    Db = decode_bounding_box_list(Bb, Cb, B)
    for bbox in chain(Da, Db):
        print(bbox)

if __name__ == '__main__':
    main()
