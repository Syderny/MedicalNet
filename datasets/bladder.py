# -*- coding: utf-8 -*-


import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class BladderDataset(Dataset):
    def __init__(self, root_dir):
        assert os.path.exists(root_dir), "root_dir: {} doesn't exist".format(root_dir)
        
        self.root_dir = root_dir
        self.patient_list = os.listdir(root_dir)
        
    
    def __len__(self):
        return len(self.patient_list)
    
    
    def __getitem__(self, index):
        patient = self.patient_list[index]
        patient_dir = os.path.join(self.root_dir, patient)
        image_paths = [os.path.join(patient_dir, x) for x in os.listdir(patient_dir)]
        image_array = [cv2.imread(x)[..., 0] for x in image_paths]
        
        image_array = np.array(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype(np.float32)
        
        label = int(patient[-1])
        
        return image_array, label
        