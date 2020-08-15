import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

import os
from PIL import Image

import matplotlib.pyplot as plt

class Edges2Shoes(Dataset):
    def __init__(self, root, transform, transform_mask, mode='train'):
        self.root = root
        self.transform = transform
        self.transform_mask = transform_mask

        self.mode = mode
        
        data_dir = os.path.join(root, mode)
        self.file_list = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.mode, self.file_list[idx])
        img = Image.open(img_path)
        W, H = img.size[0], img.size[1]
        #input data 768*256, each size = 256*256  input,ground_truth,mask
        data = img.crop((0, 0, int(W / 3), H))
        ground_truth = img.crop((int(W / 3), 0, int(2*W / 3), H))

        mask = img.crop((int(2*W / 3), 0, W, H))

        #data.save("data.jpg")
        #ground_truth.save("ground_truth.jpg")
        #mask.save("mask.jpg")


        data = self.transform(data)  #torch.Size([3, 128, 128])
        ground_truth = self.transform(ground_truth) #torch.Size([3, 128, 128])
        mask = self.transform_mask(mask) #mask unused Normalize  torch.Size([3, 128, 128])
        mask = mask.int() #Keep mask  1 or 0

        
        #for test png
        #mask_test = Image.open(r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\sky_finder\home\mihail\mypages\rpmihail\skyfinder\images\65\65.png")
        #width, height = mask_test.size
        #mask_test = Transforms.ToTensor()(mask_test).unsqueeze(0)
        #print(width, height) 

        #print(mask.shape)
        #print(data.shape)
        #print(mask)
        #print(torch.sum(mask))
        #print(sum(mask[0][0]))
        
        



        return (data, ground_truth,mask)

def data_loader(root, batch_size=1, shuffle=True, img_size=128, mode='train'):    
    transform = Transforms.Compose([Transforms.Scale((img_size, img_size)),
                                    Transforms.ToTensor(),
                                    Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))
                                   ])
    transform_mask = Transforms.Compose([Transforms.Scale((img_size, img_size)),
                                    Transforms.ToTensor()                                    
                                   ])                               
    
    dset = Edges2Shoes(root, transform,transform_mask, mode=mode)
    
    if batch_size == 'all':
        batch_size = len(dset)
        
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=0,
                                          drop_last=True)
    dlen = len(dset)
    
    return dloader, dlen