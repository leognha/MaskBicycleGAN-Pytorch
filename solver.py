import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
# for content feature extract  (import a pretrained vgg model)
import torchvision.models.vgg as models

from dataloader import data_loader
import model
import util

import os

'''
    < mse_loss >
    Calculate mean squared error loss

    * Parameters
    score : Output of discriminator
    target : 1 for real and 0 for fake
'''
def mse_loss(score, target=1):
    dtype = type(score)
    
    if target == 1:
        label = util.var(torch.ones(score.size()), requires_grad=False)
    elif target == 0:
        label = util.var(torch.zeros(score.size()), requires_grad=False)
    
    criterion = nn.MSELoss()
    loss = criterion(score, label)
    
    return loss

'''
    < mae_mask_criterion >
    Calculate mean average error mask loss

    * Parameters
    torch.mul : element-wise
'''
def mae_mask_criterion(in_, target, mask):
    #code from PTGan origin tenor=([1, 128, 128]),but here is ([3, 128, 128])
    #print mask.get_shape(), target.get_shape()
    #D_mask = torch.cat([mask,mask],0)
    #D_mask = torch.cat([D_mask,mask],0)
    #return torch.mean(torch.mul((in_-target)**2,D_mask))
    #return tf.reduce_mean((in_-target)**2)
    loss = torch.mean(torch.mul((in_-target)**2,mask))
    return loss

'''
    < L1_loss >
    Calculate L1 loss

    * Parameters
    pred : Output of network
    target : Ground truth
'''
def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

'''
    < Content_loss_vgg16 >
    Calculate L2 loss from vgg16 feature extraction

    * Parameters
    pred : Output of network
    target : Ground truth
'''
def Content_loss_vgg16(pred, target):
    
    #vgg16 = models.vgg16(pretrained=True)
    #pred = vgg16.features[:3](pred)
    #target = vgg16.features[:3](target)

    return torch.mean((pred - target)**2)

'''
    < make_model / >
    import vgg16 model to extract feature

    origin code from : https://blog.csdn.net/Geek_of_CSDN/article/details/84343971
'''
def make_model():
    model=models.vgg16(pretrained=True).features[:28]	# 其实就是定位到第28层，对照着上面的key看就可以理解
    model=model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model

def extract_feature(model,tensor):
    model.eval()		# 必须要有，不然会影响特征提取结果
    
    #img=Image.open(imgpath)		# 读取图片
    #img=img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    #tensor=img_to_tensor(img)	# 将图片转化成tensor
    #tensor=tensor.cuda()	# 如果只是在cpu上跑的话要将这行去掉
    
    result=model(Variable(tensor))
    #result_npy=result.data.cpu().numpy()	# 保存的时候一定要记得转成cpu形式的，不然可能会出错
    
    #return result_npy[0]
    return result


def lr_decay_rule(epoch, start_decay=100, lr_decay=100):
    decay_rate = 1.0 - (max(0, epoch - start_decay) / float(lr_decay))
    return decay_rate

class Solver():
    def __init__(self, root='data/edges2shoes', result_dir='result', weight_dir='weight', load_weight=False,
                 batch_size=2, test_size=20, test_img_num=5, img_size=128, num_epoch=100, save_every=1000,
                 lr=0.0002, beta_1=0.5, beta_2=0.999, lambda_kl=0.01, lambda_img=10, lambda_z=0.5, z_dim=8):
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        # Data type(Can use GPU or not?)
        torch.cuda.set_device(self.device)
        self.dtype = torch.cuda.FloatTensor
        if torch.cuda.is_available() is False:
            self.dtype = torch.FloatTensor
        


        # Data loader for training
        self.dloader, dlen = data_loader(root=root, batch_size=batch_size, shuffle=True, 
                                         img_size=img_size, mode='train')

        # Data loader for test
        self.t_dloader, _ = data_loader(root=root, batch_size=test_size, shuffle=False, 
                                        img_size=img_size, mode='val')

        # Models
        # D_cVAE is discriminator for cVAE-GAN(encoded vector z).
        # D_cLR is discriminator for cLR-GAN(random vector z).
        # Both of D_cVAE and D_cLR has two discriminators which have different output size((14x14) and (30x30)).
        # Totally, we have for discriminators now.
        self.D_cVAE = model.Discriminator().type(self.dtype)
        self.D_cLR = model.Discriminator().type(self.dtype)
        self.G = model.Generator(z_dim=z_dim).type(self.dtype)
        self.E = model.Encoder(z_dim=z_dim).type(self.dtype)




        # Optimizers
        self.optim_D_cVAE = optim.Adam(self.D_cVAE.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_D_cLR = optim.Adam(self.D_cLR.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_E = optim.Adam(self.E.parameters(), lr=lr, betas=(beta_1, beta_2))
        
        # Optiminzer lr scheduler
        #self.optim_D_scheduler = optim.lr_scheduler.LambdaLR(self.optim_D, lr_lambda=lr_decay_rule)
        #self.optim_G_scheduler = optim.lr_scheduler.LambdaLR(self.optim_G, lr_lambda=lr_decay_rule)
        #self.optim_E_scheduler = optim.lr_scheduler.LambdaLR(self.optim_E, lr_lambda=lr_decay_rule)

        # fixed random_z for test
        self.fixed_z = util.var(torch.randn(test_size, test_img_num, z_dim))
        
        # Some hyperparameters
        self.z_dim = z_dim
        self.lambda_kl = lambda_kl
        self.lambda_img = lambda_img
        self.lambda_z = lambda_z

        # Extra things
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.load_weight = load_weight
        self.test_img_num = test_img_num
        self.img_size = img_size
        self.start_epoch = 0
        self.num_epoch = num_epoch
        self.save_every = save_every
        
    '''
        < show_model >
        Print model architectures
    '''
    def show_model(self):
        print('=========================== Discriminator for cVAE ===========================')
        print(self.D_cVAE)
        print('=============================================================================\n\n')
        print('=========================== Discriminator for cLR ===========================')
        print(self.D_cLR)
        print('=============================================================================\n\n')
        print('================================= Generator =================================')
        print(self.G)
        print('=============================================================================\n\n')
        print('================================== Encoder ==================================')
        print(self.E)
        print('=============================================================================\n\n')
        
    '''
        < set_train_phase >
        Set training phase
    '''
    def set_train_phase(self):
        self.D_cVAE.train()
        self.D_cLR.train()
        self.G.train()
        self.E.train()
        
    '''
        < load_pretrained >
        If you want to continue to train, load pretrained weight
    '''
    def load_pretrained(self):
        self.D_cVAE.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D_cVAE.pkl')))
        self.D_cLR.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D_cLR.pkl')))
        self.G.load_state_dict(torch.load(os.path.join(self.weight_dir, 'G.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(self.weight_dir, 'E.pkl')))
        
        log_file = open('log.txt', 'r')
        line = log_file.readline()
        self.start_epoch = int(line)
        
    '''
        < save_weight >
        Save weight
    '''
    def save_weight(self, epoch=None):
        if epoch is None:
            d_cVAE_name = 'D_cVAE.pkl'
            d_cLR_name = 'D_cLR.pkl'
            g_name = 'G.pkl'
            e_name = 'E.pkl'
        else:
            d_cVAE_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cVAE.pkl')
            d_cLR_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cLR.pkl')
            g_name = '{epochs}-{name}'.format(epochs=str(epoch), name='G.pkl')
            e_name = '{epochs}-{name}'.format(epochs=str(epoch), name='E.pkl')
            
        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cVAE_name))
        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cLR_name))
        torch.save(self.G.state_dict(), os.path.join(self.weight_dir, g_name))
        torch.save(self.E.state_dict(), os.path.join(self.weight_dir, e_name))
    
    '''
        < all_zero_grad >
        Set all optimizers' grad to zero 
    '''
    def all_zero_grad(self):
        self.optim_D_cVAE.zero_grad()
        self.optim_D_cLR.zero_grad()
        self.optim_G.zero_grad()
        self.optim_E.zero_grad()
        
    '''
        < train >
        Train the D_cVAE, D_cLR, G and E 
    '''
    def train(self):
        if self.load_weight is True:
            self.load_pretrained()
        
        self.set_train_phase()
        self.show_model()
        #import pretrained vgg16 model
        vgg_model=make_model()

        
        # Training Start!
        for epoch in range(self.start_epoch, self.num_epoch):
            for iters, (img, ground_truth, mask) in enumerate(self.dloader):
                # img(2, 3, 128, 128) : Two images in Domain A. One for cVAE and another for cLR. 
                # ground_truth(2, 3, 128, 128) : Two images Domain B. One for cVAE and another for cLR.
                img, ground_truth = util.var(img), util.var(ground_truth)
                mask = util.var(mask)
                # Seperate data for cVAE_GAN(using encoded z) and cLR_GAN(using random z)
                cVAE_data = {'img' : img[0].unsqueeze(dim=0), 'ground_truth' : ground_truth[0].unsqueeze(dim=0), 'mask' : mask[0].unsqueeze(dim=0)}
                cLR_data = {'img' : img[1].unsqueeze(dim=0), 'ground_truth' : ground_truth[1].unsqueeze(dim=0), 'mask' : mask[1].unsqueeze(dim=0)}


                ''' ----------------------------- 1. Train D ----------------------------- '''
                #######################  < Step 1. D loss in cVAE-GAN >  #######################

                # Encoded latent vector
                mu, log_variance = self.E(cVAE_data['ground_truth'])
                std = torch.exp(log_variance / 2)
                random_z = util.var(torch.randn(1, self.z_dim))
                encoded_z = (random_z * std) + mu

                # Generate fake image
                fake_img_cVAE = self.G(cVAE_data['img'], encoded_z)
                
                real_pair_cVAE = torch.cat([cVAE_data['img'], cVAE_data['ground_truth']], dim=1)
                fake_pair_cVAE = torch.cat([cVAE_data['img'], fake_img_cVAE], dim=1)
                
                real_d_cVAE_1, real_d_cVAE_2 = self.D_cVAE(real_pair_cVAE)
                fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_pair_cVAE.detach())
                
                D_loss_cVAE_1 = mse_loss(real_d_cVAE_1, 1) + mse_loss(fake_d_cVAE_1, 0) # Small patch loss
                D_loss_cVAE_2 = mse_loss(real_d_cVAE_2, 1) + mse_loss(fake_d_cVAE_2, 0) # Big patch loss
                
                #######################  < Step 2. D loss in cLR-GAN >  #######################
                
                # Generate fake image
                # Generated img using 'cVAE' data will be used to train D_'cLR'
                fake_img_cLR = self.G(cVAE_data['img'], random_z)
                
                real_pair_cLR = torch.cat([cLR_data['img'], cLR_data['ground_truth']], dim=1)
                fake_pair_cLR = torch.cat([cVAE_data['img'], fake_img_cLR], dim=1)
                
                # A_cVAE = Domain A image for cVAE, A_cLR = Domain A image for cVAE
                # B_cVAE = Domain B image for cVAE, B_cLR = Domain B image for cVAE
                
                # D_cVAE has to discriminate [A_cVAE, B_cVAE] vs [A_cVAE, G(A_cVAE, encoded_z)]
                # D_cLR has to discriminate [A_cLR, B_cLR] vs [A_cVAE, G(A_cVAE, random_z)]
                
                # This helps to generate more diverse images
                real_d_cLR_1, real_d_cLR_2 = self.D_cLR(real_pair_cLR)
                fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_pair_cLR.detach())
                
                D_loss_cLR_1 = mse_loss(real_d_cLR_1, 1) + mse_loss(fake_d_cLR_1, 0) # Small patch loss
                D_loss_cLR_2 = mse_loss(real_d_cLR_2, 1) + mse_loss(fake_d_cLR_2, 0) # Big patch loss

                D_loss = D_loss_cVAE_1 + D_loss_cVAE_2 + D_loss_cLR_1 + D_loss_cLR_2

                # Update D
                self.all_zero_grad()
                D_loss.backward()
                self.optim_D_cVAE.step()
                self.optim_D_cLR.step()

                ''' ----------------------------- 2. Train G & E ----------------------------- '''
                ########### < Step 1. GAN loss to fool discriminator (cVAE_GAN and cLR_GAN) > ###########
                
                # Encoded latent vector
                mu, log_variance = self.E(cVAE_data['ground_truth'])
                std = torch.exp(log_variance / 2)
                random_z = util.var(torch.randn(1, self.z_dim))
                encoded_z = (random_z * std) + mu

                # Generate fake image
                fake_img_cVAE = self.G(cVAE_data['img'], encoded_z)
                fake_pair_cVAE = torch.cat([cVAE_data['img'], fake_img_cVAE], dim=1)
                
                # Fool D_cVAE
                fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_pair_cVAE)

                GAN_loss_cVAE_1 = mse_loss(fake_d_cVAE_1, 1) # Small patch loss
                GAN_loss_cVAE_2 = mse_loss(fake_d_cVAE_2, 1) # Big patch 
                
                #L2 mask loss 
                Mask_loss = mae_mask_criterion(fake_img_cVAE, cVAE_data['img'], cVAE_data['mask'])
                #Mask_loss = mae_mask_criterion(fake_img_cVAE, cVAE_data['ground_truth'], cVAE_data['mask'])

                #content loss 
                fake_img_cVAE_feature = extract_feature(vgg_model, fake_img_cVAE)
                cVAE_data_feature = extract_feature(vgg_model, cVAE_data['img'])
                Content_loss = Content_loss_vgg16(fake_img_cVAE_feature, cVAE_data_feature)

                # Random latent vector and generate fake image
                random_z = util.var(torch.randn(1, self.z_dim))
                fake_img_cLR = self.G(cLR_data['img'], random_z)
                fake_pair_cLR = torch.cat([cLR_data['ground_truth'], fake_img_cLR], dim=1)
                
                # Fool D_cLR
                fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_pair_cLR)

                GAN_loss_cLR_1 = mse_loss(fake_d_cLR_1, 1) # Small patch loss
                GAN_loss_cLR_2 = mse_loss(fake_d_cLR_2, 1) # Big patch loss


                G_GAN_loss = GAN_loss_cVAE_1 + GAN_loss_cVAE_2 + GAN_loss_cLR_1 + GAN_loss_cLR_2 + Mask_loss + 0.5*Content_loss


                ################# < Step 2. KL-divergence with N(0, 1) (cVAE-GAN) > #################
                
                # See http://yunjey47.tistory.com/43 or Appendix B in the paper for details
                KL_div = self.lambda_kl * torch.sum(0.5 * (mu ** 2 + torch.exp(log_variance) - log_variance - 1))

                #### < Step 3. Reconstruction of ground truth image (|G(A, z) - B|) (cVAE-GAN) > ####
                img_recon_loss = self.lambda_img * L1_loss(fake_img_cVAE, cVAE_data['ground_truth'])

                EG_loss = G_GAN_loss + KL_div + img_recon_loss
                self.all_zero_grad()
                EG_loss.backward(retain_graph=True) # retain_graph=True for the next step 3. Train ONLY G
                self.optim_E.step()
                self.optim_G.step()

                ''' ----------------------------- 3. Train ONLY G ----------------------------- '''
                ##### < Step 1. Reconstrution of random latent code (|E(G(A, z)) - z|) (cLR-GAN) > #####
                
                # This step should update only G.
                # See https://github.com/junyanz/BicycleGAN/issues/5 for details.
                mu, log_variance = self.E(fake_img_cLR)
                z_recon_loss = L1_loss(mu, random_z)

                z_recon_loss = self.lambda_z * z_recon_loss

                self.all_zero_grad()
                z_recon_loss.backward()
                self.optim_G.step()

                log_file = open('log.txt', 'w')
                log_file.write(str(epoch))
                
                # Print error, save intermediate result image and weight
                #if iters % self.save_every == 0:
                #    print('[Epoch : %d / Iters : %d] => D_loss : %f / G_GAN_loss : %f / KL_div : %f / img_recon_loss : %f / z_recon_loss : %f'\
                #          %(epoch, iters, D_loss.data[0], G_GAN_loss.data[0], KL_div.data[0], img_recon_loss.data[0], z_recon_loss.data[0]))
                if iters % self.save_every == 0:
                    print('[Epoch : %d / Iters : %d] => D_loss : %f / G_GAN_loss : %f / KL_div : %f / img_recon_loss : %f / z_recon_loss : %f / Mask_loss : %f / Content_loss : %f' \
                          % (epoch, iters, D_loss.data, G_GAN_loss.data, KL_div.data, img_recon_loss.data,z_recon_loss.data,Mask_loss.data,Content_loss.data))
                    # Save intermediate result image
                    if os.path.exists(self.result_dir) is False:
                        os.makedirs(self.result_dir)

                    result_img = util.make_img(self.t_dloader, self.G, self.fixed_z,img_num=self.test_img_num, img_size=self.img_size)

                    img_name = '{epoch}_{iters}.png'.format(epoch=epoch, iters=iters)
                    img_path = os.path.join(self.result_dir, img_name)

                    torchvision.utils.save_image(result_img, img_path, nrow=self.test_img_num+1)

                    # Save intermediate weight
                    if os.path.exists(self.weight_dir) is False:
                        os.makedirs(self.weight_dir)
                    
                    self.save_weight()
                    
            # Save weight at the end of every epoch
            self.save_weight(epoch=epoch)
