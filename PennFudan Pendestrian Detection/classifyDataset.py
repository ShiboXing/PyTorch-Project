import os
import numpy as np
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
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        mask = Image.open(mask_path)
        
        #convert the PIL image into a numpy array
        mask = np.array(mask)
        '''print(np.array(Image.open(img_path)))'''

        obj_ids = np.unique(mask)
        #first id is the background
        obj_ids = obj_ids[1:]
        masks = obj_ids[:,None,None] == mask 
       
        testMask = np.arange(9).reshape((3,3))
        testMasks = testMask < np.array([4,5])[:, None, None]
        #print(testMask, testMasks)

        testMask = np.arange(9).reshape((3,3))
        testMasks = testMask == np.array([3,4])[:, None, None]

        #get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            #similar to find() in matlab
            pos = np.where(masks[i])
            #since masks is 2d, then coordinates are two 2d, thus getting 0 and 1 in pos
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,),dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])


        '''
        print(boxes)
        print(boxes[:,3],boxes[:,1],boxes[:,3] - boxes[:,1])
        '''
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = img_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target







pfd = PennFudanDataset('PennFudanPed',None)
mask = Image.open('./PennFudanPed/PNGImages/FudanPed00010.png')
pfd[0]




#mask.show()
'''
arr = [1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 17, 20]
arr1 = [[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12]]
n_arr = np.array(arr)
n_arr1 = np.array(arr1)
print(n_arr[...,5],'-',n_arr[6])
print(n_arr1[2,1],'-',n_arr1[1])
#print(n_arr[:, None, None])
print(n_arr1[:, None, None])
'''

# PFD = PennFudanDataset('./','')

'''
print(np.array([4,5]),'@')
print(np.array([4,5])[None,:],'@')
print(np.array([4,5])[None,:,None],'@')
'''

'''
print(np.array([0,1,2])[:,None,None])
print(testMask)
print(testMask == np.array([0,1,2])[:,None,None])
'''

'''
testMask = np.arange(9).reshape((3,3))
testMasks = testMask == np.array([[0,1,2],[10,10,10],[6,7,8]])[:, None, None]
print(testMasks, testMask)
print(np.array([[0,1,2],[10,10,10],[6,7,8]]))
'''

'''
test = np.array([[[0,1,1],[0,0,0],[0,0,0]],[[0,1,0],[0,0,0],[0,0,0]]])
pos = np.where(test)
print(test, pos)
'''