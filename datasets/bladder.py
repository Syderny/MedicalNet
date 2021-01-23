# -*- coding: utf-8 -*-


import os

import cv2
import numpy as np
from torch import unsqueeze
from torch.utils.data import Dataset
from torchvision import transforms


class BladderDataset(Dataset):
    def __init__(self, root_dir, sets):
        assert os.path.exists(root_dir), "root_dir: {} doesn't exist".format(root_dir)
        
        self.root_dir = root_dir
        self.sets = sets
        self.patient_list = os.listdir(root_dir)
        
    
    def __len__(self):
        return len(self.patient_list)
    
    
    def __getitem__(self, index):
        patient = self.patient_list[index]
        patient_dir = os.path.join(self.root_dir, patient)
        image_paths = [os.path.join(patient_dir, x) for x in os.listdir(patient_dir)]
        image_array = [cv2.imread(x)[..., 0] for x in image_paths]
        
        image_array = np.array(image_array)
        image_array = image_array.astype(np.float32)

        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.sets.input_H, self.sets.input_W))
        ])

        image_array = self.__itensity_normalize_one_volume__(image_array)
        image_array = transform1(image_array)
        image_array = unsqueeze(image_array, dim=0)
        
        label = int(patient[-1])
        
        return image_array, label
        
    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out