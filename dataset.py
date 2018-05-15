import torchvision.transforms as transforms
import json
from collections import defaultdict
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
import os
import numpy as np

from util import rescale_bounding_box
from util import center_bounding_box

import pyximport; pyximport.install()
from bounding_box import encode_bounding_box_list_many_to_one
from bounding_box import encode_bounding_box_list_one_to_one

class COCO(Dataset):
    """
    COCO Dataset that supports bounding boxes as outputs
    
    Parameters
    ----------
    
    folder : str
        folder where the COCO dataset is located.
        the folder should contain (for instance):
            annotations/instances_train2014.json
            annotations/captions_train2014.json
            annotations/instances_valid2014.json
            annotations/captions_valid2014.json
            train2014/
            valid2014/
    anchor_list :
        list of anchors to use for each scale
    split : 'train2014' or 'valid2014'
        which dataset split to use.
    transform : Transform
        pytorch transform to apply to images
    """
    def __init__(self, folder, anchor_list, split='train2014', transform=None):
        self.imgs_folder = os.path.join(folder, split)
        self.anchor_list = anchor_list
        self.annotations_folder = os.path.join(folder, 'annotations')
        self.split = split
        self.transform = transform
        self._load_annotations()
        self.bbox_encodings = {}
        self.background_class_id = 0

    def _load_annotations(self):
        A = json.load(open(os.path.join(self.annotations_folder, 'instances_{}.json'.format(self.split))))
        B = json.load(open(os.path.join(self.annotations_folder, 'captions_{}.json'.format(self.split))))
        
        class_id_name = {a['id']: a['name'] for a in A['categories']}

        index_to_image_id = {}
        image_id_to_filename = {}
        for b in B['images']:
            image_id_to_filename[b['id']] = b['file_name']
        rng = np.random.RandomState(42)
        keys = list(image_id_to_filename.keys())
        rng.shuffle(keys)
        
        index_to_filename = {i: image_id_to_filename[k] for i, k in enumerate(keys)}
        image_id_to_index = {k: i for i, k in enumerate(keys)}
        
        classes = set()
        for a in A['annotations']:
            cat = class_id_name[a['category_id']]
            classes.add(cat)
        self.filenames = index_to_filename
        self.classes = sorted(list(classes))
        # IMPORTANT
        # a new class is added here, the class 0
        # class 0 is the background class
        # from class 1 to len(classes) is the rest
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.class_to_idx['background'] = 0
        self.idx_to_class = {i: c for c, i in class_to_idx.items()}
        boxes = defaultdict(list)
        for a in A['annotations']:
            bbox = a['bbox']
            cat = class_id_name[a['category_id']]
            cat_idx = self.class_to_idx[cat]
            boxes[image_id_to_index[a['image_id']]].append((bbox, cat_idx))
        self.boxes = boxes

    def _load_bbox_encodings(self, bboxes):
        Y = encode_bounding_box_list_many_to_one(
            bboxes, 
            self.anchor_list, 
            background_class_id=0)
        return Y

    def _load(self, i):
        filename = self.filenames[i]
        x = default_loader(os.path.join(self.imgs_folder, 'img', filename))
        from_size = x.size
        if self.transform:
            x = self.transform(x)
        to_size = x.size(2), x.size(1)
        boxes = self.boxes[i]
        boxes = [(center_bounding_box(box), class_id) for box, class_id in boxes]
        boxes = [(rescale_bounding_box(box, from_size, to_size), self.class_to_idx[cat]) for box, cat in boxes]
        return x, boxes

    def __getitem__(self, i):
        x, bboxes = self._load(i)
        # caching
        if i in self.bbox_encodings:
            e = self.bbox_encodings[i]
        else:
            e = self._load_bbox_encodings(bboxes)
        return x, bboxes, e

    def __len__(self):
        return len(self.filenames)

class VOC(Dataset):
    def __init__(self, folder, anchor_list, split='train', transform=None):
        self.folder = folder
        self.anchor_list = anchor_list
        self.split = split
        self.transform = transform
        self._load_annotations()
        self.bbox_encodings = {}
        self.background_class_id = 0

    def _load_annotations(self):
        from voc_utils import get_all_obj_and_box, list_image_sets
        import random
        random.seed(42)
        anns = get_all_obj_and_box(self.split, self.folder)
        random.shuffle(anns)
        self.filenames = [fname for fname, bboxes in anns]
        self.boxes = [bboxes for fname, bboxes in anns]
        self.classes = sorted(list_image_sets())
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.class_to_idx['background'] = 0
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def _load_bbox_encodings(self, bboxes):
        Y = encode_bounding_box_list_many_to_one(
            bboxes, 
            self.anchor_list, 
            background_class_id=0)
        return Y

    def _load(self, i):
        filename = self.filenames[i]
        x = default_loader(os.path.join(filename))
        from_size = x.size
        if self.transform:
            x = self.transform(x)
        to_size = x.size(2), x.size(1)
        boxes = self.boxes[i]
        boxes = [(center_bounding_box(box), class_id) for box, class_id in boxes]
        boxes = [(rescale_bounding_box(box, from_size, to_size), self.class_to_idx[cat]) for box, cat in boxes]
        return x, boxes

    def __getitem__(self, i):
        x, bboxes = self._load(i)
        # caching
        if i in self.bbox_encodings:
            e = self.bbox_encodings[i]
        else:
            e = self._load_bbox_encodings(bboxes)
        return x, bboxes, e

    def __len__(self):
        return len(self.filenames)


class SubSample:

    def __init__(self, dataset, nb):
        self.dataset = dataset
        self.nb = nb
        self.classes = dataset.classes
        self.background_class_id = dataset.background_class_id
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = dataset.idx_to_class

    def __getitem__(self, i):
        return self.dataset[i]
        
    def __len__(self):
        return self.nb

if __name__ == '__main__':
    from bounding_box import build_anchors
    aspect_ratios = [[1, 2, 1.2]] * 6
    anchor_list = [
        build_anchors(scale=0.2, feature_map_size=37, aspect_ratios=aspect_ratios[0]),   
        build_anchors(scale=0.34, feature_map_size=19, aspect_ratios=aspect_ratios[1]),   
        build_anchors(scale=0.48, feature_map_size=10, aspect_ratios=aspect_ratios[2]),   
        build_anchors(scale=0.62, feature_map_size=5, aspect_ratios=aspect_ratios[3]),   
        build_anchors(scale=0.76, feature_map_size=3, aspect_ratios=aspect_ratios[4]),   
        build_anchors(scale=0.90, feature_map_size=1, aspect_ratios=aspect_ratios[5]),   
    ]
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    dataset = VOC('voc', anchor_list, split='train', transform=transform)
    x, b, e = dataset[0]
    print(x.size())
    print(b)
    print(e)
