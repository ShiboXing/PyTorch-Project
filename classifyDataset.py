import os
import numpy
import torch
from PIL import Image

class PennFudanDataset(object):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        #load all images, sorting them to 
        #ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        #load imafges ad masks
        img_path = os.path.join(root, "PNGImages")
        mask_path = os.path.join(root, "PedMasks")
        img = Image.open(img_path).convert("RGB")
        
        mask = Image.open(mask_path)
        
        mask.show()

        obj_ids = np.unique(mask)

        obj_ids = obj_ids[1:]



PennFudanDataset = PennFudanDataset('./PennFudanPed','./PennFudanPed')
mask = Image.open('./PennFudanPed/PNGImages/PennPed00011.png')
mask.show()

