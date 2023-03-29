import torch, os, time, utils
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from utils import log_sum_exp, save_tensor_images
from torch.autograd import Variable
import torch.optim as optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def reg_loss(featureT,fea_mean, fea_logvar):
    
    fea_reg = reparameterize(fea_mean, fea_logvar)
    fea_reg = fea_mean.repeat(featureT.shape[0],1)
    loss_reg = torch.mean((featureT - fea_reg).pow(2))
    # print('loss_reg',loss_reg)
    return loss_reg

def attack_acc(fake,iden,E,):
    
    eval_prob = E(utils.low2high(fake))[-1]
    
    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
    
    cnt, cnt5 = 0, 0
    bs = fake.shape[0]
    # print('correct id')
    for i in range(bs):
        gt = iden[i].item()
        if eval_iden[i].item() == gt:
            cnt += 1
            # print(gt)
        _, top5_idx = torch.topk(eval_prob[i], 5)
        if gt in top5_idx:
            cnt5 += 1
    return cnt*100.0/bs, cnt5*100.0/bs

def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu

def find_criterion(used_loss):
    criterion = None
    if used_loss=='logit_loss':
        criterion = nn.NLLLoss().to(device)
        print('criterion:{}'.format(used_loss))
    elif used_loss=='cel':
        criterion = nn.CrossEntropyLoss().to(device)    
        print('criterion',criterion)
    else:
        print('criterion:{}'.format(used_loss))
    return criterion

def get_act_reg(train_loader,T,device,Nsample=5000):
    all_fea = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader): # batchsize =100
            # print(data.shape)
            if batch_idx*len(data) > Nsample:
                break
            data  = data.to(device)
            fea,_ = T(data)
            if batch_idx == 0:
                all_fea = fea
            else:
                all_fea = torch.cat((all_fea,fea))
    fea_mean = torch.mean(all_fea,dim=0)
    fea_logvar = torch.std(all_fea,dim=0)
    
    print(fea_mean.shape, fea_logvar.shape, all_fea.shape)
    return fea_mean,fea_logvar

def iden_loss(T,fake, iden, used_loss,criterion,fea_mean=0, fea_logvar=0,lam=0.1):
    Iden_Loss = 0
    loss_reg = 0
    for tn in T:
      
        feat,out = tn(fake)
        if used_loss == 'logit_loss': #reg only with the target classifier, reg is randomly from distribution
            if Iden_Loss ==0:                
                loss_sdt =  criterion(out, iden)
                loss_reg = lam*reg_loss(feat,fea_mean[0], fea_logvar[0]) #reg only with the target classifier

                Iden_Loss = Iden_Loss + loss_sdt  
            else:                
                loss_sdt =  criterion(out, iden)
                Iden_Loss = Iden_Loss + loss_sdt

        else:
            loss_sdt = criterion(out, iden)
            Iden_Loss = Iden_Loss + loss_sdt

    Iden_Loss = Iden_Loss/len(T) + loss_reg
    return Iden_Loss



def dist_inversion(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, \
                   iter_times=1500, clip_range=1.0, improved=False, num_seeds=5, \
                   used_loss='cel', prefix='', random_seed=0, save_img_dir='',fea_mean=0, \
                   fea_logvar=0, lam=0.1, clipz=False):
    
    iden = iden.view(-1).long().to(device)
    criterion = find_criterion(used_loss)
    bs = iden.shape[0]
    
    G.eval() 
    D.eval()
    E.eval()
    
    #NOTE
    mu = Variable(torch.zeros(bs, 100), requires_grad=True)
    log_var = Variable(torch.ones(bs, 100), requires_grad=True)
    
    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)
    outputs_z = "{}_iter_{}_{}_dis.npy".format(prefix, random_seed, iter_times-1)
    
    if not os.path.exists(outputs_z):
        outputs_z = "{}_iter_{}_{}_dis".format(prefix, random_seed, 0)
        outputs_label = "{}_iter_{}_{}_label".format(prefix, random_seed, 0)
        np.save(outputs_z,{"mu":mu.detach().cpu().numpy(),"log_var":log_var.detach().cpu().numpy()})
        np.save(outputs_label,iden.detach().cpu().numpy())
            
        for i in range(iter_times):
            z = reparameterize(mu, log_var)
            if clipz==True:
                z =  torch.clamp(z,-clip_range,clip_range).float()
            fake = G(z)

            if improved == True:
                _, label =  D(fake)
            else:
                label = D(fake)
                    
            for p in params:
                if p.grad is not None:
                    p.grad.data.zero_()
            Iden_Loss = iden_loss(T,fake, iden, used_loss, criterion, fea_mean, fea_logvar, lam)

            if improved:
                Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
            else:
                Prior_Loss = - label.mean()

            Total_Loss = Prior_Loss + lamda * Iden_Loss
           
            Total_Loss.backward()
            solver.step()

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if (i+1) % 300 == 0:
                outputs_z = "{}_iter_{}_{}_dis".format(prefix, random_seed, i)
                outputs_label = "{}_iter_{}_{}_label".format(prefix, random_seed, i)
                np.save(outputs_z,{"mu":mu.detach().cpu().numpy(),"log_var":log_var.detach().cpu().numpy()})
                np.save(outputs_label,iden.detach().cpu().numpy())
        
                with torch.no_grad():
                    z = reparameterize(mu, log_var)
                    if clipz==True:
                        z =  torch.clamp(z,-clip_range, clip_range).float()
                    fake_img = G(z.detach())
                    eval_prob = E(utils.low2high(fake_img))[-1]
                    
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
                    save_tensor_images(fake_img, save_img_dir + '{}.png'.format(i+1))
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
                    
                        
        outputs_z = "{}_iter_{}_{}_dis".format(prefix, random_seed, iter_times)
        outputs_label = "{}_iter_{}_{}_label".format(prefix, random_seed, iter_times)
        np.save(outputs_z,{"mu":mu.detach().cpu().numpy(),"log_var":log_var.detach().cpu().numpy()})
        np.save(outputs_label,iden.detach().cpu().numpy())
       
def inversion(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, \
                iter_times=1500, clip_range=1, improved=False, num_seeds=5, \
                used_loss='cel', prefix='', save_img_dir='', fea_mean=0, \
                fea_logvar=0, lam=0.1, istart=0, same_z=''):
    
    iden = iden.view(-1).long().to(device)
    criterion = find_criterion(used_loss)
    bs = iden.shape[0]
    
    G.eval()
    D.eval()
    E.eval()

    for random_seed in range(istart, num_seeds, 1):
        outputs_z = "{}_iter_{}_{}_z.npy".format(prefix, random_seed, iter_times-1)
    
        if not os.path.exists(outputs_z):
            tf = time.time()
            if same_z=='': #no prior z
                z = torch.randn(bs, 100).to(device).float()
            else:
                z_path = "{}_iter_{}_{}_z.npy".format(same_z, random_seed, 0)
                print('load z ', z_path)
                z = torch.from_numpy(np.load(z_path)).to(device).float() 
                print('z',z)
            # exit() 
            z.requires_grad = True
            v = torch.zeros(bs, 100).to(device).float()
                
            outputs_z = "{}_iter_{}_{}_z".format(prefix, random_seed, 0)
            outputs_label = "{}_iter_{}_label".format(prefix, random_seed, 0)
            np.save(outputs_z,z.detach().cpu().numpy())
            np.save(outputs_label,iden.detach().cpu().numpy())
            
            for i in range(iter_times):
                fake = G(z)
                if improved == True:
                    _, label =  D(fake)
                else:
                    label = D(fake)
                
                if z.grad is not None:
                    z.grad.data.zero_()

                if improved:
                    Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
                else:
                    Prior_Loss = - label.mean()

                Iden_Loss = iden_loss(T,fake, iden, used_loss, criterion, fea_mean, fea_logvar, lam)

                Total_Loss = Prior_Loss + lamda*Iden_Loss

                Total_Loss.backward()
                
                v_prev = v.clone()
                gradient = z.grad.data
                v = momentum*v - lr*gradient
                z = z + ( - momentum*v_prev + (1 + momentum)*v)
                z = torch.clamp(z.detach(), -clip_range, clip_range).float()
                z.requires_grad = True

                Prior_Loss_val = Prior_Loss.item()
                Iden_Loss_val = Iden_Loss.item()

                if (i+1) % 300 == 0:
                    outputs_z = "{}_iter_{}_{}_z".format(prefix, random_seed, i)
                    outputs_label = "{}_iter_{}_{}_label".format(prefix, random_seed, i)
                    np.save(outputs_z, z.detach().cpu().numpy())
                    np.save(outputs_label, iden.detach().cpu().numpy())
                    with torch.no_grad():
                        fake_img = G(z.detach())
                        
                        eval_prob = E(utils.low2high(fake_img))[-1]
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
                        print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.3f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
    
