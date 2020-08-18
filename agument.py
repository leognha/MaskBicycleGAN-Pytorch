import torch
import torchvision

from dataloader_agu import data_loader
import model
import util

import os
import numpy as np
import argparse


def interpolation_by_one_dim(img_num=6, z_dim=8):


    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    interpolated_z = torch.FloatTensor(img_num, z_dim).type(dtype)
    #print(interpolated_z.size())

   
    #first_z = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0, 0,  -1.5]]).type(dtype)   #-2
    #last_z = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0,  0,  1.5]]).type(dtype)    #2

    #cloud less/more
    interpolated_z[0] = torch.FloatTensor([[ -1, 0, 0, 0, 0, 0, 0, 0]]).type(dtype)
    interpolated_z[1] = torch.FloatTensor([[ 1, 0, 0, 0, 0, 0, 0, 0]]).type(dtype)
    #dark/light
    interpolated_z[2] = torch.FloatTensor([[ 0, 0, 0, 0, 0, 0, 0, -0.8]]).type(dtype)
    interpolated_z[3] = torch.FloatTensor([[ 0, 0, 0, 0, 0, 0, 0, 0.8]]).type(dtype)
    #blue/yellow
    interpolated_z[4] = torch.FloatTensor([[ 0, 0, 0, 0, 0, -1, 0, 0]]).type(dtype)
    interpolated_z[5] = torch.FloatTensor([[ 0, 0, 0, 0, 0, 1, 0, 0]]).type(dtype)



    
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

    #z = util.var(interpolation_by_one_dim(img_num=img_num, z_dim=z_dim))

    #randn
    z = util.var(torch.randn(img_num, z_dim))
    
    return z


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
    img, _ , _, _= iter_dloader.next()
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
                                img_size=128, mode=args.mode)

    #data_file_path = os.path.join(args.root, args.mode)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Data type(Can use GPU or not?)
    torch.cuda.set_device(device)

    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    if args.epoch is not None:
        weight_name = '{epoch}-G.pkl'.format(epoch=args.epoch)
    else:
        weight_name = 'G.pkl'
        
    weight_path = os.path.join(args.weight_dir, weight_name)
    G = model.Generator(z_dim=16).type(dtype)
    G.load_state_dict(torch.load(weight_path))
    G.eval()

    weight_path2 = os.path.join(args.weight2_dir, weight_name)
    G2 = model.Generator(z_dim=2).type(dtype)
    G2.load_state_dict(torch.load(weight_path2))
    G2.eval()
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epoch is None:
        args.epoch = 'latest'
    
    i = 0
    for iters, (img, ground_truth, mask, file_name) in enumerate(dloader):
        img = util.var(img.type(dtype))
        #mask = util.var(mask.type(dtype))
        #one = torch.ones([1, 3, 128, 128])
        #one = util.var(one.type(dtype))



        z = make_z(img_num=args.img_num, z_dim=16)
        z2 = make_z(img_num=args.img_num, z_dim=2)
        for j in range(args.img_num):
            z_ = z[j, :].unsqueeze(dim=0)
            z2_ = z2[j, :].unsqueeze(dim=0)

            out_img = G(img, z_)
            out_img2 = G2(img, z2_)

            outs_img =out_img/ 2 + 0.5
            outs_img2 =out_img2/ 2 + 0.5

            img_name = '{filenames}_{style}.png'.format(filenames = file_name[0], style = j)
            img_name2 = '{filenames}_{style}_1.png'.format(filenames = file_name[0], style = j)
            #print(img_name)
            #mask_name = '{filenames}_{style}.png'.format(filenames = filenames[i], style = j)

            img_path = os.path.join(args.result_dir, img_name)
            img_path2 = os.path.join(args.result_dir, img_name2)
            #mask_path = os.path.join(args.mask_dir, mask_name)


            # for FID SCORE
            #fileDir = '/home/leognha/Desktop/seg-model/MedicalImage_Project02_Segmentation/data/split1/train1.25'

            #pathDir = os.listdir(fileDir)

            #if img_name in pathDir:
            #    torchvision.utils.save_image(outs_img2, img_path2)


            torchvision.utils.save_image(outs_img, img_path)
            torchvision.utils.save_image(outs_img2, img_path2)

            #torchvision.utils.save_image(mask_, mask_path)
        i= i+1
        print(i,'in split3')
    
    print('origin number:',len(os.listdir(os.path.join(args.root, args.mode))))
    print('agu number:',len(os.listdir(args.result_dir)))

        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('--root', type=str, default='data/edges2shoes/',
                        #default='data/edges2shoes',
                        help='Data location')

    parser.add_argument('--mode', type=str, default='compareD',                    
                        help='Data location')

    parser.add_argument('--result_dir', type=str, default='result/compareD',
                        help='Ouput images location')
    parser.add_argument('--mask_dir', type=str, default='agu0.5/masks',
                        help='Ouput images location,unuse now')
    parser.add_argument('--weight_dir', type=str, default='weight/z=16',
                        help='Trained weight location of generator. pkl file location')

    parser.add_argument('--weight2_dir', type=str, default='weight/z=2',
                        help='Trained weight location of generator. pkl file location')

    parser.add_argument('--img_num', type=int, default=2,
                        help='Generated images number per one input image')
    parser.add_argument('--epoch', type=int,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')

    args = parser.parse_args()
    main(args)