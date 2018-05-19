import torchvision.transforms as transforms
import json
from collections import defaultdict
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
import os
import numpy as np
import PIL

import pyximport; pyximport.install()
from bounding_box import encode_bounding_box_list_many_to_one
from bounding_box import encode_bounding_box_list_one_to_one
from bounding_box import rescale_bounding_box
from bounding_box import center_bounding_box
from bounding_box import normalize_bounding_box
from bounding_box import decode_bounding_box_list
from voc_utils import get_all_obj_and_box, list_image_sets

class DetectionDataset(Dataset):

    def _load_bbox_encodings(self, bboxes):
        Y = encode_bounding_box_list_many_to_one(
            bboxes, 
            self.anchor_list, 
            background_class_id=self.background_class_id,
            iou_threshold=self.iou_threshold)
        return Y

    def _load(self, i):
        filename = self.filenames[i]
        boxes = self.boxes[i]
        x = default_loader(os.path.join(self.imgs_folder, 'img', filename))
        # use the full image or randomly sample a patch
        """
        u = self.rng.uniform(0, 1)
        if u <= 0.5:
            scale = self.rng.uniform(0.3, 1.0)
            ar = self.rng.uniform(0.5, 1.0)
            boxes_ = []
            while len(boxes_) == 0:
                x_, crop_box = _random_patch(self.rng, x, scale, ar)
                boxes_ = [(box, cat) for box, cat in boxes if box_in_box(box, crop_box)]
                bx, by, bw, bh = crop_box
                boxes_ = [((x - bx, y - by, w, h), cat) for (x, y, w, h), cat in boxes_]
            boxes = boxes_
            x = x_
        """
        # flip
        """
        u = self.rng.uniform(0, 1)
        if u <= 0.5:
            x = x.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            boxes = [((x.size[0] - bx - bw, by, bw, bh), cat) for (bx, by, bw, bh), cat in boxes] 
        """
        # apply transform
        from_size = x.size
        x = self.transform(x)
        to_size = x.size(2), x.size(1)
        boxes = [(rescale_bounding_box(box, from_size, to_size), self.class_to_idx[cat]) for box, cat in boxes]
        boxes = [(normalize_bounding_box(box, to_size), cat) for box, cat in boxes]
        boxes = [(center_bounding_box(box), class_id) for box, class_id in boxes]
        return x, boxes

    def __getitem__(self, i):
        x, bboxes = self._load(i)
        e = self._load_bbox_encodings(bboxes)
        return x, bboxes, e

    def __len__(self):
        return len(self.boxes)



class COCO(DetectionDataset):
    def __init__(self, folder, anchor_list, split='train2014', iou_threshold=0.5, classes=None, transform=None, random_state=42):
        self.imgs_folder = os.path.join(folder, split)
        self.anchor_list = anchor_list
        self.annotations_folder = os.path.join(folder, 'annotations')
        self.split = split
        self.transform = transform
        self.classes = classes
        self.bbox_encodings = {}
        self.background_class_id = 0
        self.iou_threshold = iou_threshold
        self.rng = np.random.RandomState(random_state)
        self._load_annotations()
    
    def _load_annotations(self):
        A = json.load(open(os.path.join(self.annotations_folder, 'instances_{}.json'.format(self.split))))
        B = json.load(open(os.path.join(self.annotations_folder, 'captions_{}.json'.format(self.split))))
        
        class_id_name = {a['id']: a['name'] for a in A['categories']}

        index_to_image_id = {}
        image_id_to_filename = {}
        for b in B['images']:
            image_id_to_filename[b['id']] = b['file_name']
        
        keys = list(image_id_to_filename.keys())
        self.rng.shuffle(keys)
        index_to_filename = {i: image_id_to_filename[k] for i, k in enumerate(keys)}
        image_id_to_index = {k: i for i, k in enumerate(keys)}
        
        if self.classes:
            classes = set(self.classes)
        else:
            classes = set()
            for a in A['annotations']:
                cat = class_id_name[a['category_id']]
                classes.add(cat)
        self.classes = sorted(list(classes))
        # IMPORTANT
        # a new class is added here, the class 0
        # class 0 is the background class
        # from class 1 to len(classes) is the rest
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.class_to_idx['background'] = 0
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        B = defaultdict(list)
        F = {}
        for a in A['annotations']:
            bbox = a['bbox']
            cat = class_id_name[a['category_id']]
            if cat in classes:
                B[image_id_to_index[a['image_id']]].append((bbox, cat))
        indexes = list(index_to_filename.keys())
        self.boxes = [B[ind] for ind in indexes if len(B[ind]) > 0]
        self.filenames = [index_to_filename[ind] for ind in indexes if len(B[ind]) > 0]
    

class VOC(DetectionDataset):
    def __init__(self, folder, anchor_list, which='VOC2007', split='train', iou_threshold=0.5, classes=None, transform=None, random_state=42):
        self.folder = folder # root folder, should contain VOC2007 and/or VOC2012
        self.which = which
        self.anchor_list = anchor_list
        self.split = split
        self.transform = transform
        self.background_class_id = 0
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.rng = np.random.RandomState(random_state)
        self._load_annotations()

    def _load_annotations(self):
        voc2007 = os.path.join(self.folder, 'VOC2007')
        voc2012 = os.path.join(self.folder, 'VOC2012')
        if self.which == 'VOC2007':
            paths = [voc2007]
        elif self.which == 'VOC2012':
            paths = [voc2012]
        elif self.which == 'VOC0712':
            paths = [voc2007, voc2012]
        else:
            raise ValueError('which should be voc2007 or voc2012 or voc0712')
        anns = []
        for path in paths:
            anns += get_all_obj_and_box(self.split, path, classes=self.classes)
        classes = set()
        for fname, bboxes in anns:
            for (x, y, w, h), class_name in bboxes:
                classes.add(class_name)
        classes = list(classes)
        if self.classes:
            assert set(classes) == set(self.classes)
        self.rng.shuffle(anns)
        self.filenames = [fname for fname, bboxes in anns]
        self.boxes = [bboxes for fname, bboxes in anns]
        self.classes = sorted(classes)
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.class_to_idx['background'] = 0
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}


def box_in_box(box_small, box_big):
    bsx, bsy, bsw, bsh = box_small
    bbx, bby, bbw, bbh = box_big
    if not (bsx >= bbx and bsx + bsw <= bbx + bbw):
        return False
    if not (bsy >= bby and bsy + bsh <= bby + bbh):
        return False
    return True


def _random_patch(rng, im, scale, aspect_ratio):
    w, h = im.size
    wcrop = int(scale * w)
    hcrop = min(int(wcrop / aspect_ratio), h) 
    xmin, ymin = rng.randint(0, w - wcrop + 1), rng.randint(0, h - hcrop + 1)
    xmax = xmin + wcrop
    ymax = ymin + hcrop
    return im.crop((xmin, ymin, xmax, ymax)), (xmin, ymin, wcrop, hcrop)


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
    aspect_ratios = [[1, 2, 3, 1/2, 1/3]] * 6
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
    classes = ['bottle', 'cup', 'wine glass', 'bowl']
    dataset = COCO(
        'data/coco', 
        anchor_list, 
        split='train2014',
        iou_threshold=0.5,
        classes=classes,
        transform=transform
    )
    A = []
    for i in range(len(dataset)):
        bb = dataset.boxes[i]
        print(i, len(dataset))
        for b in bb:
            (x, y, w, h), _ = b
            A.append(w/h)
    from sklearn.cluster import KMeans
    clus = KMeans(n_clusters=6)
    A = np.array(A).reshape((-1, 1))
    clus.fit(A)
    print(clus.cluster_centers_)
