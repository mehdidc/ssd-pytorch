import random
import json
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
import os
import numpy as np
import PIL

import pyximport; pyximport.install()
from bounding_box import encode_bounding_box_list_many_to_one
from bounding_box import rescale_bounding_box
from bounding_box import center_bounding_box
from bounding_box import normalize_bounding_box
from bounding_box import box_in_box
from voc_utils import get_all_obj_and_box

class DetectionDataset(Dataset):

    def _load_bbox_encodings(self, bboxes):
        Y = encode_bounding_box_list_many_to_one(
            bboxes, 
            self.anchor_list, 
            background_class_id=self.background_class_id,
            iou_threshold=self.iou_threshold,
            variance=self.variance,
        )
        return Y

    def _load(self, i):
        filename = self.filenames[i]
        boxes = self.boxes[i]
        assert len(boxes) > 0, i
        x = default_loader(filename)
        # randomly sample a patch
        
        da_params = self.data_augmentation_params
        if da_params is None:
            da_params = {}
        u = random.uniform(0, 1)
        if u <= da_params.get('patch_proba', 0):
            min_scale = da_params.get('min_scale', 0.1)
            max_scale = da_params.get('max_scale', 1)
            min_ar = da_params.get('min_aspect_ratio', 0.5)
            max_ar = da_params.get('max_aspect_ratio', 2)
            max_nb_trials = da_params.get('nb_trials', 50)
            scale = random.uniform(min_scale, max_scale)
            ar = random.uniform(min_ar, max_ar)
            boxes_ = []
            nb_trials = 0
            while len(boxes_) == 0 and nb_trials < max_nb_trials:
                x_, crop_box = _random_patch(self.rng, x, scale, ar)
                boxes_ = [(box, cat) for box, cat in boxes if box_in_box(box, crop_box)]
                bx, by, bw, bh = crop_box
                boxes_ = [((x - bx, y - by, w, h), cat) for (x, y, w, h), cat in boxes_]
                nb_trials += 1
            if len(boxes_) > 0:
                boxes = boxes_
                x = x_
        assert len(boxes) > 0, i
        # flip
        u = random.uniform(0, 1)
        if u <= da_params.get('flip_proba', 0):
            x = x.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            boxes = [((x.size[0] - bx - bw, by, bw, bh), cat) for (bx, by, bw, bh), cat in boxes] 
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
    def __init__(self, folder='data/coco', anchor_list=[], 
                 split='train2014', iou_threshold=0.5, 
                 data_augmentation_params=None,
                 classes=None, transform=None, 
                 variance=[0.1, 0.1, 0.2, 0.2],
                 random_state=42):
        self.folder = folder
        self.anchor_list = anchor_list
        self.annotations_folder = os.path.join(folder, 'annotations')
        self.split = split
        self.transform = transform
        self.classes = classes
        self.background_class_id = 0
        self.iou_threshold = iou_threshold
        self.data_augmentation_params = data_augmentation_params
        self.rng = np.random.RandomState(random_state)
        self.variance = variance
        self._load_annotations()
    
    def _load_annotations(self):
        A = json.load(open(os.path.join(self.annotations_folder, 'instances_{}.json'.format(self.split))))
        B = json.load(open(os.path.join(self.annotations_folder, 'captions_{}.json'.format(self.split))))
        class_id_name = {a['id']: a['name'] for a in A['categories']}
        image_id_to_filename = {}
        for b in B['images']:
            image_id_to_filename[b['id']] = b['file_name']
        keys = list(image_id_to_filename.keys())
        keys = sorted(keys)
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
        for a in A['annotations']:
            bbox = a['bbox']
            cat = class_id_name[a['category_id']]
            if cat in classes:
                B[image_id_to_index[a['image_id']]].append((bbox, cat))
        indexes = list(index_to_filename.keys())
        self.boxes = [B[ind] for ind in indexes if len(B[ind]) > 0]
        self.filenames = [index_to_filename[ind] for ind in indexes if len(B[ind]) > 0]
        self.filenames = [os.path.join(self.folder, self.split,  f) for f in self.filenames]


class WIDER(DetectionDataset):
    
    def __init__(self, folder='data/wider', 
                 anchor_list=[], 
                 split='train', 
                 iou_threshold=0.5, 
                 data_augmentation_params=None, 
                 transform=None, 
                 variance=[0.1, 0.1, 0.2, 0.2],
                 random_state=42):
        self.folder = folder
        self.anchor_list = anchor_list
        self.split = split
        self.transform = transform
        self.background_class_id = 0
        self.iou_threshold = iou_threshold
        self.data_augmentation_params = data_augmentation_params
        self.rng = np.random.RandomState(random_state)
        self.variance = variance
        self._load_annotations()

    def _load_annotations(self):
        images_folder = os.path.join(
            self.folder, 
            'WIDER_{}'.format(self.split), 
            'images'
        )
        annotation_file = os.path.join(
            self.folder, 
            'wider_face_split', 
            'wider_face_{}_bbx_gt.txt'.format(self.split)
        )
        anns = []
        with open(annotation_file) as fd:
            while True:
                f = fd.readline().strip()
                if f == '':
                    break
                filename = os.path.join(images_folder, f)
                nb_boxes = int(fd.readline().strip())
                bboxes = []
                for i in range(nb_boxes):
                    line = fd.readline().strip()
                    toks = line.split(' ')
                    x, y, w, h, *rest = toks
                    x = float(x)
                    y = float(y)
                    w = float(w)
                    h = float(h)
                    print(x, y, w, h)
                    box = (x, y, w, h), 'person'
                    bboxes.append(box)
                anns.append((filename, bboxes))
        self.rng.shuffle(anns)
        self.filenames = [fname for fname, bboxes in anns]
        self.boxes = [bboxes for fname, bboxes in anns]
        self.classes = ['person']
        self.class_to_idx = {'background': 0, 'person': 1}
        self.idx_to_class = {0: 'background', 1: 'person'}


class VOC(DetectionDataset):
    def __init__(self, folder='data/voc', anchor_list=[], 
                 which='VOC2007', split='train', 
                 iou_threshold=0.5, data_augmentation_params=None, 
                 classes=None, transform=None, 
                 variance=[0.1, 0.1, 0.2, 0.2],
                 random_state=42):
        self.folder = folder # root folder, should contain VOC2007 and/or VOC2012
        self.which = which
        self.anchor_list = anchor_list
        self.split = split
        self.transform = transform
        self.background_class_id = 0
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.data_augmentation_params = data_augmentation_params
        self.rng = np.random.RandomState(random_state)
        self.variance = variance
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
        anns = sorted(anns)
        anns = [(fname, bboxes) for fname, bboxes in anns if len(bboxes) > 0]
 
        self.rng.shuffle(anns)
        self.filenames = [fname for fname, bboxes in anns]
        self.boxes = [bboxes for fname, bboxes in anns]
        self.classes = sorted(classes)
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.class_to_idx['background'] = 0
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}


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
        nb = min(len(dataset), nb)
        self.dataset = dataset
        self.nb = nb
        self.classes = dataset.classes
        self.background_class_id = dataset.background_class_id
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = dataset.idx_to_class
        self.transform = dataset.transform

    def __getitem__(self, i):
        return self.dataset[i]
        
    def __len__(self):
        return self.nb
