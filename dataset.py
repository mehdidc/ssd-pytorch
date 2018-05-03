import json
from collections import defaultdict
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
import os
import numpy as np

from util import rescale_bounding_box
from util import center_bounding_box


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
    split : 'train2014' or 'valid2014'
        which dataset split to use.
    transform : Transform
        pytorch transform to apply to images
    """
    def __init__(self, folder, split='train2014', transform=None):
        self.imgs_folder = os.path.join(folder, split)
        self.annotations_folder = os.path.join(folder, 'annotations')
        self.split = split
        self.transform = transform
        self._load_annotations()

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
        
        boxes = defaultdict(list)
        classes = set()
        for a in A['annotations']:
            bbox = a['bbox']
            cat = a['category_id']
            classes.add(cat)
            boxes[image_id_to_index[a['image_id']]].append((bbox, cat))
        self.filenames = index_to_filename
        self.boxes = boxes
        self.classes = list(classes)
        # IMPORTANT
        # a new class is added here, the class 0
        # class 0 is the background class
        # from class 1 to len(classes) is the rest
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.idx_to_class = {i + 1: cl for i, cl in enumerate(self.classes)}
        self.idx_to_class[0] = 'background'
        self.class_name = {
            i: (class_id_name[cl] if cl in class_id_name else 'background')
            for i, cl in self.idx_to_class.items()} 

    def __getitem__(self, i):
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
    
    def __len__(self):
        return len(self.filenames)


class SubSample:

    def __init__(self, dataset, nb):
        self.dataset = dataset
        self.nb = nb
        self.classes = dataset.classes
        self.class_name = dataset.class_name

    def __getitem__(self, i):
        return self.dataset[i]
        
    def __len__(self):
        return self.nb



if __name__ == '__main__':
    dataset = COCO('.')
    print(dataset[0][1])
