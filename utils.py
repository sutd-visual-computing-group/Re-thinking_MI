import torch.nn.init as init
import os, models.facenet as facenet, sys
import json, time, random, torch
from models import classify
from models.classify import *
from models.discri import *
from models.generator import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
import dataloader
from torch.autograd import grad

device = "cuda"

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

def init_dataloader(args, file_path, batch_size=64, mode="gan", iterator=False):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

  
    data_set = dataloader.ImageFolder(args, file_path, mode)

    if iterator:
        data_loader = torch.utils.data.DataLoader(data_set,
                                batch_size=batch_size,
                                shuffle=shuffle_flag,
                                drop_last=True,
                                num_workers=0,
                                pin_memory=True).__iter__()
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                batch_size=batch_size,
                                shuffle=shuffle_flag,
                                drop_last=True,
                                num_workers=2,
                                pin_memory=True)
        interval = time.time() - tf
        print('Initializing data loader took %ds' % interval)
    
    return data_set, data_loader

def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')

def save_tensor_images(images, filename, nrow = None, normalize = True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)


def get_deprocessor():
    # resize 112,112
    proc = []
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    return transforms.Compose(proc)

def low2high(img):
    # 0 and 1, 64 to 112
    bs = img.size(0)
    proc = get_deprocessor()
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]
    
    img = img.cuda()
    return img


def get_model(attack_name, classes):
    if attack_name.startswith("VGG16"):
        T = classify.VGG16(classes)
    elif attack_name.startswith("IR50"):
        T = classify.IR50(classes)
    elif attack_name.startswith("IR152"):
        T = classify.IR152(classes)
    elif attack_name.startswith("FaceNet64"):
        T = facenet.FaceNet64(classes)
    else:
        print("Model doesn't exist")
        exit()

    T = torch.nn.DataParallel(T).cuda()
    return T

def get_augmodel(model_name, nclass, path_T=None, dataset='celeba'):
    if model_name=="VGG16":
        model = VGG16(nclass)   
    elif model_name=="FaceNet":
        model = FaceNet(nclass)
    elif model_name=="FaceNet64":
        model = FaceNet64(nclass)
    elif model_name=="IR152":
        model = IR152(nclass)
    elif model_name =="efficientnet_b0":
        model = classify.EfficientNet_b0(nclass)   
    elif model_name =="efficientnet_b1":
        model = classify.EfficientNet_b1(nclass)   
    elif model_name =="efficientnet_b2":
        model = classify.EfficientNet_b2(nclass)  

    model = torch.nn.DataParallel(model).cuda()
    if path_T is not None: 
        
        ckp_T = torch.load(path_T)        
        model.load_state_dict(ckp_T['state_dict'], strict=True)
    return model


def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

# define "soft" cross-entropy with pytorch tensor operations
def softXEnt (input, target):
    targetprobs = nn.functional.softmax (target, dim = 1)
    logprobs = nn.functional.log_softmax (input, dim = 1)
    return  -(targetprobs * logprobs).sum() / input.shape[0]

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
    
def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y, DG):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_GAN(dataset, gan_type, gan_model_dir, n_classes, z_dim, target_model):

    G = Generator(z_dim)
    if gan_type == True:
        D = MinibatchDiscriminator(n_classes=n_classes)
    else:
        D = DGWGAN(3)

    if gan_type == True:
        path = os.path.join(os.path.join(gan_model_dir, dataset), target_model)
        path_G = os.path.join(path, "improved_{}_G.tar".format(dataset))
        path_D = os.path.join(path, "improved_{}_D.tar".format(dataset))
    else:
        path = os.path.join(gan_model_dir, dataset)
        path_G = os.path.join(path, "{}_G.tar".format(dataset))
        path_D = os.path.join(path, "{}_D.tar".format(dataset)) 

    print('path_G',path_G)
    print('path_D',path_D)

    G = torch.nn.DataParallel(G).to(device)
    D = torch.nn.DataParallel(D).to(device)
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=True)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=True)
  
    return G, D


def get_attack_model(args, args_json, eval_mode=False):
    now = datetime.now() # current date and time
    
    if not eval_mode:
        log_file = "invertion_logs_{}_{}.txt".format(args.loss,now.strftime("%m_%d_%Y_%H_%M_%S"))
        utils.Tee(os.path.join(args.log_path, log_file), 'w')
    
    n_classes=args_json['dataset']['n_classes']
    
    
    model_types_ = args_json['train']['model_types'].split(',')
    checkpoints = args_json['train']['cls_ckpts'].split(',')

    G, D = get_GAN(args_json['dataset']['name'],gan_type=args.improved_flag, 
                    gan_model_dir=args_json['train']['gan_model_dir'],
                    n_classes=n_classes,z_dim=100,target_model=model_types_[0])

    dataset = args_json['dataset']['name']
    cid = args.classid.split(',')
    # target and student classifiers
    for i in range(len(cid)):
        id_ = int(cid[i])
        model_types_[id_] = model_types_[id_].strip()
        checkpoints[id_] = checkpoints[id_].strip()
        print('Load classifier {} at {}'.format(model_types_[id_], checkpoints[id_]))
        model = get_augmodel(model_types_[id_],n_classes,checkpoints[id_],dataset)
        model = model.to(device)
        model = model.eval()
        if i==0:
            targetnets = [model]
        else:
            targetnets.append(model)
    
        # p_reg 
        if args.loss=='logit_loss':
            if model_types_[id_] == "IR152" or model_types_[id_]=="VGG16" or model_types_[id_]=="FaceNet64": 
                #target model
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"], '{}_{}_p_reg.pt'.format(dataset,model_types_[id_])) #'./p_reg/{}_{}_p_reg.pt'.format(dataset,model_types_[id_])
            else:
                #aug model
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"], '{}_{}_{}_p_reg.pt'.format(dataset,model_types_[0],model_types_[id_])) #'./p_reg/{}_{}_{}_p_reg.pt'.format(dataset,model_types_[0],model_types_[id_])
            # print('p_reg',p_reg)
            if not os.path.exists(p_reg):
                _, dataloader_gan = init_dataloader(args_json, args_json['dataset']['gan_file_path'], 50, mode="gan")
                from attack import get_act_reg
                fea_mean_,fea_logvar_ = get_act_reg(dataloader_gan,model,device)
                torch.save({'fea_mean':fea_mean_,'fea_logvar':fea_logvar_},p_reg)
            else:
                fea_reg = torch.load(p_reg)
                fea_mean_ = fea_reg['fea_mean']
                fea_logvar_ = fea_reg['fea_logvar']
            if i == 0:
                fea_mean = [fea_mean_.to(device)]
                fea_logvar = [fea_logvar_.to(device)]
            else:
                fea_mean.append(fea_mean_)
                fea_logvar.append(fea_logvar_)
            # print('fea_logvar_',i,fea_logvar_.shape,fea_mean_.shape)
            
        else:
            fea_mean,fea_logvar = 0,0

    # evaluation classifier
    E = get_augmodel(args_json['train']['eval_model'],n_classes,args_json['train']['eval_dir'])    
    E.eval()
    G.eval()
    D.eval()

    return targetnets, E, G, D, n_classes, fea_mean, fea_logvar
