import torch
import numpy as np
import os
from metrics.fid import concatenate_list, gen_samples
from utils import load_json, save_tensor_images


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_shortest_dist(fea_target,fea_fake):
    shortest_dist = 0
    pdist = torch.nn.PairwiseDistance(p=2)

    fea_target = torch.from_numpy(fea_target).to(device)
    fea_fake = torch.from_numpy(fea_fake).to(device)
    # print('---fea_fake.shape[0]',fea_fake.shape[0])
    for i in range(fea_fake.shape[0]):
        dist = pdist(fea_fake[i,:], fea_target)
        
        min_d = min(dist)
        
        # print('--KNN dist',min_d)
        shortest_dist = shortest_dist + min_d*min_d
    # print('--KNN dist',shortest_dist)

    return shortest_dist

def run_KNN(target_dir, fake_dir):
    knn = 0
    target = np.load(target_dir,allow_pickle=True)  
    fake = np.load(fake_dir,allow_pickle=True)  
    target_fea = target.item().get('fea')    
    target_y = target.item().get('label')
    fake_fea = fake.item().get('fea')
    fake_y = fake.item().get('label')

    fake_fea = concatenate_list(fake_fea)
    fake_y = concatenate_list(fake_y)
    
    N = fake_fea.shape[0]
    for id in range(300):
        id_f = fake_y == id
        id_t = target_y == id
        fea_f = fake_fea[id_f,:]
        fea_t = target_fea[id_t]
        
        shorted_dist = find_shortest_dist(fea_t,fea_f)
        knn = knn + shorted_dist  
        
    return knn/N

def eval_KNN(G, E, save_dir, KNN_real_path, args):
    
    fea_path, _ = gen_samples(G, E, save_dir, args.improved_flag)

    fea_path = fea_path + 'full.npy'

    knn = run_KNN(KNN_real_path, fea_path)
    print("KNN:{:.3f} ".format(knn))
    return knn
