from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json

class CUS_Dataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 7  #Sara: 7 body parts and background #21

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(CUS_Dataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root)
        print(self.split)
        if self.split == "val":
            file_list = os.path.join("{}/val.txt".format(self.root))
        elif self.split == 'train_supervised':
            file_list = os.path.join("{}/sup_eval.txt".format(self.root))
            #file_list = os.path.join("{}/sup_train.txt".format(self.root))
            #file_list = os.path.join("{}/weak_sup.txt".format(self.root)) 
        elif self.split == "train_unsupervised":
            file_list = os.path.join("{}/unsup_eval.txt".format(self.root))
            #file_list = os.path.join("{}/unsup_train.txt".format(self.root))
            #file_list = os.path.join("{}/weak_unsup.txt".format(self.root))
        elif self.split == "train_unsupervised_sequence":
            file_list = os.path.join("{}/unsup_eval_3seq.txt".format(self.root))
            #file_list = os.path.join("{}/unsup_train_5seq.txt".format(self.root))
        else:
            raise ValueError(f"Invalid split name {self.split}")

        if self.split == "train_unsupervised_sequence":
            self.files = tuple([tuple(line.rstrip().split(' ')) for line in tuple(open(file_list, "r"))])
            self.labels = tuple(["/data/sara/CCT/body_part_data/data/sara_blank_img.png" for i in range(len(self.files))])
        else:
            file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
            self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_paths = self.files[index] #os.path.join(self.root, self.files[index][1:])
        if type(image_paths) is not tuple:
            return self._load_data_single(index, image_paths)
        else:
            return ([self._load_data_single(index, path) for path in image_paths],)
                

    def _load_data_single(self, index, image_path):
        img_obj = Image.open(image_path)
        width, height = img_obj.size

        new_h = 400
        #h_percent = new_h / float(height)
        #new_w = int(float(width) * float(h_percent))
        new_w = 598
        image = np.asarray(img_obj.resize((new_w, new_h)), dtype=np.float32)
        # WASimage = np.asarray(Image.open(image_path), dtype=np.float32)
        # WAS image_id = self.files[index].split("/")[-1].split(".")[0]
        last_part = image_path.split("/")[-1].split(".")[-1]
        image_id = image_path.split("/")[-1].replace("." + last_part, '')
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        else:
            label_path = self.labels[index] #os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path).resize((new_w, new_h), Image.NEAREST), dtype=np.int32)
        #label = np.asarray(Image.open(label_path), dtype=np.int32)
        
        #print(image_id)
        #print(image_path)
        #print(label_path)
        #print(image_path, ":", image.shape, label_path, ":", label.shape, image_id) 
        return image, label, image_id

class CUS_loader(BaseDataLoader):
    def __init__(self, kwargs):
        ## sara Calculated mean and std mean: 115.23671735292434, 101.76216371197306, 91.09968687628187
        ## std: 54.82798581119987, 51.122941378859736, 50.94759578729262
        self.MEAN =  [0.45, 0.4, 0.35] # voc[0.485, 0.456, 0.406] 
        self.STD =  [0.21, 0.2, 0.2] # voc[0.229, 0.224, 0.225]  
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = CUS_Dataset(**kwargs)

        super(CUS_loader, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
