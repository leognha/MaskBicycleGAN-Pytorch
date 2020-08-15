import torch
import torchvision

from dataloader import data_loader
import model
import util

import os
import numpy as np
import argparse


def interpolation_by_one_dim(img_num=4, z_dim=8):
    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    interpolated_z = torch.FloatTensor(img_num, z_dim).type(dtype)
    #print(interpolated_z.size())

   
    #first_z = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0, 0,  -1.5]]).type(dtype)   #-2
    #last_z = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0,  0,  1.5]]).type(dtype)    #2

    interpolated_z[0] = torch.FloatTensor([[ -0.5, 0, 0, 0, 0, 0, 0, 0]]).type(dtype)
    interpolated_z[1] = torch.FloatTensor([[ 1, 0, 0, 0, 0, 0, 0, 0]]).type(dtype)
    interpolated_z[2] = torch.FloatTensor([[ 0, 0, 0, 0, 0, 0, -0.8, 0]]).type(dtype)
    interpolated_z[3] = torch.FloatTensor([[ 0, 0, 0, 0, 0, 0, 0.8, 0]]).type(dtype)
    
    return interpolated_z

'''
    < make_z >
    Make latent code
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
    sample_type : random or interpolation or interpolation_by_every_dim
'''
def make_z(img_num, z_dim=8):

    z = util.var(interpolation_by_one_dim(img_num=img_num, z_dim=z_dim))
    
    return z

def get_files_name():
    dst = []
    input_files_path = 'data/edges2shoes/test'
    #input_files_path = '/home/leognha/Desktop/seg-model/MedicalImage_Project02_Segmentation/data/imgs'
    tmp = os.listdir(input_files_path)
    for i in tmp:
        dst.append(i.split('.')[0])
    #print("files_path: {}".format(dst[0]))
    return dst


'''
    < make_img >
    Generate images.
    
    * Parameters
    dloader : Dataloader
    G : Generator
    z : Random latent code with size of (N, img_num, z_dim)
    img_size : Image size. Now only 128 is available.
    img_num : Generated images number per one input image.
'''
def make_img(dloader, G, z, img_size=128):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    iter_dloader = iter(dloader)
    img, _ , _= iter_dloader.next()
    img_num = z.size(1)

    N = img.size(0)    
    img = util.var(img.type(dtype))

    result_img = torch.FloatTensor(N * (img_num + 1), 3, img_size, img_size).type(dtype)

    for i in range(N):
        # The leftmost is domain A image(Edge image)
        result_img[i * (img_num + 1)] = img[i].data

        # Generate img_num images per a domain A image
        for j in range(img_num):
            img_ = img[i].unsqueeze(dim=0)
            z_ = z[i, j, :].unsqueeze(dim=0)
            
            out_img = G(img_, z_)
          
            result_img[i * (img_num + 1) + j + 1] = out_img.data

            #save image every input
            #img_name = '{type}_{epoch}_{z}.png'.format(type=args.sample_type, epoch=args.epoch,z=z_)
            #img_path = os.path.join(args.result_dir, img_name)
            #torchvision.utils.save_image(out_img, img_path, nrow=args.img_num + 1, padding=4)


    result_img = result_img / 2 + 0.5
    
    return result_img
        
def main(args):    
    dloader, dlen = data_loader(root=args.root, batch_size=1, shuffle=False, 
                                img_size=128, mode='test')

    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    if args.epoch is not None:
        weight_name = '{epoch}-G.pkl'.format(epoch=args.epoch)
    else:
        weight_name = 'G.pkl'
        
    weight_path = os.path.join(args.weight_dir, weight_name)
    G = model.Generator(z_dim=8).type(dtype)
    G.load_state_dict(torch.load(weight_path))
    G.eval()
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epoch is None:
        args.epoch = 'latest'
    
    filenames = get_files_name()

    for iters, (img, ground_truth, mask) in enumerate(dloader):
        img = util.var(img.type(dtype))
        #mask = util.var(mask.type(dtype))
        one = torch.ones([1, 3, 128, 128])
        one = util.var(one.type(dtype))


        for i in range(0,dlen):
            # img_ = img.unsqueeze(dim=0)
            
            #mask_ = mask[i].unsqueeze(dim=0)
            #mask_ = one - mask_

            # Make latent code and images
            z = make_z(img_num=4, z_dim=8)
            for j in range(4):
                z_ = z[j, :].unsqueeze(dim=0)
                out_img = G(img, z_)
                outs_img =out_img/ 2 + 0.5

                img_name = '{filenames}_{style}.png'.format(filenames = filenames[i], style = j)
                print(img_name)
                #mask_name = '{filenames}_{style}.png'.format(filenames = filenames[i], style = j)

                img_path = os.path.join(args.result_dir, img_name)
                #mask_path = os.path.join(args.mask_dir, mask_name)

                torchvision.utils.save_image(outs_img, img_path)
                #torchvision.utils.save_image(mask_, mask_path)

            #result_img = make_img(dloader, G, z, img_size=128)   
            #torchvision.utils.save_image(result_img, img_path, nrow=args.img_num + 1, padding=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_type', type=str, choices=['random', 'interpolation','interpolation_by_every_dim','interpolation_by_one_dim'], default='random',
                        help='Type of sampling : \'random\' or \'interpolation\'') #interpolation_by_every_dim and interpolation_by_one_dim for test 
    parser.add_argument('--root', type=str, #default='/home/leognha/Desktop/seg-model/MedicalImage_Project02_Segmentation/data'
                        default='data/edges2shoes',help='Data location')
    parser.add_argument('--result_dir', type=str, default='agu/imgs',
                        help='Ouput images location')
    parser.add_argument('--mask_dir', type=str, default='agu/masks',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='weight',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--img_num', type=int, default=5,
                        help='Generated images number per one input image')
    parser.add_argument('--epoch', type=int,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')

    args = parser.parse_args()
    main(args)