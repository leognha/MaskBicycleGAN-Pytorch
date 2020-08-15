import torch
import torchvision

from dataloader_agu import data_loader
import model
import util

import os
import numpy as np
import argparse

'''
    < make_interpolation >
    Make linear interpolated latent code.
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
'''
def make_interpolation(n=200, img_num=8, z_dim=8):
    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    step = 1 / (img_num-1)
    alpha = torch.from_numpy(np.arange(0, 1, step))
    interpolated_z = torch.FloatTensor(n, img_num, z_dim).type(dtype)
    #print(alpha)
    for i in range(n):
        first_z = torch.randn(1, z_dim) #1*8
        last_z = torch.randn(1, z_dim)
        print(i)
        #print("first_z=")
        #print(first_z)
            
        #print("last_z=")
        #print(last_z)

        for j in range(img_num-1):
            interpolated_z[i, j] = (1 - alpha[j]) * first_z + alpha[j] * last_z

        interpolated_z[i, img_num-1] = last_z
    
    #print(max(interpolated_z))
    #print(min(interpolated_z))
    return interpolated_z

def interpolation_by_every_dim(n=200, img_num=8, z_dim=8):
    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    #step = 1 / (img_num-1)
    #alpha = torch.from_numpy(np.arange(0, 1, step))
    #print(alpha)
    interpolated_z = torch.FloatTensor(n, img_num, z_dim).type(dtype)
    #print(interpolated_z.size())

    for i in range(n):
        
        last_z = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0,  0,  0]]).type(dtype)
        
        for j in range(img_num-1):
            #first_z = [[ 0,  0, 0, 0,  0,  0,  0,  0]]
            #print(interpolated_z)
            #print(first_z)
            last_z = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0,  0,  0]]).type(dtype)
            last_z[0][j] = 1
            print(i,j)
            print(last_z)
            #first_z[1,j]=1.0
            interpolated_z[i, j] = last_z            
        interpolated_z[i, img_num-1] = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0,  0,  1]]).type(dtype)
    
    return interpolated_z

def interpolation_by_one_dim(n=200, img_num=8, z_dim=8):
    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    step = 1 / (img_num-1)
    alpha = torch.from_numpy(np.arange(0, 1, step))
    interpolated_z = torch.FloatTensor(n, img_num, z_dim).type(dtype)
    #print(interpolated_z.size())

    for i in range(n):
        first_z = torch.FloatTensor([[-1,  0, 0, 0,  0,  0,  0,  0]]).type(dtype)   #(-2)
        last_z = torch.FloatTensor([[ 1,  0, 0, 0,  0,  0,  0,  0]]).type(dtype)    #(2)
        
        for j in range(img_num-1):
            
            #first_z = [[ 0,  0, 0, 0,  0,  0,  0,  0]]
            #print(interpolated_z)
            #print(first_z)
            #last_z = torch.FloatTensor([[ 0,  0, 0, 0,  0,  0,  0,  0]]).type(dtype)
            #last_z[0][j] = 1
            
            #first_z[1,j]=1.0
            interpolated_z[i, j] = (1 - alpha[j]) * first_z + alpha[j] * last_z
            print(i,j)
            print(interpolated_z[i, j])
            #interpolated_z[i, j] = last_z            
        interpolated_z[i, img_num-1] = last_z  
    
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
def make_z(n, img_num, z_dim=8, sample_type='random'):
    if sample_type == 'random':
        z = util.var(torch.randn(n, img_num, 8))
    elif sample_type == 'interpolation':
        z = util.var(make_interpolation(n=n, img_num=img_num, z_dim=z_dim))
    elif sample_type == 'interpolation_by_every_dim':
        z = util.var(interpolation_by_every_dim(n=n, img_num=img_num, z_dim=z_dim))
    elif sample_type == 'interpolation_by_one_dim':
        z = util.var(interpolation_by_one_dim(n=n, img_num=img_num, z_dim=z_dim))
    
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
    img, _ , _,_= iter_dloader.next()
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
    dloader, dlen = data_loader(root=args.root, batch_size='all', shuffle=False, 
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
    weight_path2 = os.path.join(args.weight2_dir, weight_name)

    G = model.Generator(z_dim=8).type(dtype)
    G.load_state_dict(torch.load(weight_path))
    G.eval()

    G2 = model.Generator(z_dim=8).type(dtype)
    G2.load_state_dict(torch.load(weight_path2))
    G2.eval()
    




    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epoch is None:
        args.epoch = 'latest'
    #img_name = '{type}_{epoch}.png'.format(type=args.sample_type, epoch=args.epoch)
    img_name = '11.png'
    img_name2 = '12.png'


    img_path = os.path.join(args.result_dir, img_name)
    img_path2 = os.path.join(args.result_dir, img_name2)


    # Make latent code and images
    z = make_z(n=dlen, img_num=args.img_num, z_dim=8, sample_type=args.sample_type)

    result_img = make_img(dloader, G, z, img_size=128)   
    result_img2 = make_img(dloader, G2, z, img_size=128)   


    torchvision.utils.save_image(result_img, img_path, nrow=args.img_num + 1, padding=4)
    torchvision.utils.save_image(result_img2, img_path2, nrow=args.img_num + 1, padding=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_type', type=str, choices=['random', 'interpolation','interpolation_by_every_dim','interpolation_by_one_dim'], default='interpolation',
                        help='Type of sampling : \'random\' or \'interpolation\'') #interpolation_by_every_dim and interpolation_by_one_dim for test 
    parser.add_argument('--root', type=str, default='data/edges2shoes', 
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='PPT',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='weight/origin',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--weight2_dir', type=str, default='weight/0.5',
                        help='Trained weight location of generator. pkl file location')

    parser.add_argument('--img_num', type=int, default=9,
                        help='Generated images number per one input image')
    parser.add_argument('--epoch', type=int,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')

    args = parser.parse_args()
    main(args)