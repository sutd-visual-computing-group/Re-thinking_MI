from utils import *
from models.classify import *
from models.generator import *
from models.discri import *
import torch
import numpy as np

from attack import attack_acc
import statistics 

from metrics.fid import concatenate_list, gen_samples


device = torch.torch.cuda.is_available()

def accuracy(fake_dir, E):
    
    aver_acc, aver_acc5, aver_std, aver_std5 = 0, 0, 0, 0

    N = 5
    E.eval()
    for i in range(1):      
        all_fake = np.load(fake_dir+'full.npy',allow_pickle=True)  
        all_imgs = all_fake.item().get('imgs')
        all_label = all_fake.item().get('label')

        # calculate attack accuracy
        with torch.no_grad():
            N_succesful = 0
            N_failure = 0

            for random_seed in range(len(all_imgs)):
                if random_seed % N == 0:
                    res, res5 = [], []
                    
                #################### attack accuracy #################
                fake = all_imgs[random_seed]
                label = all_label[random_seed]

                label = torch.from_numpy(label)
                fake = torch.from_numpy(fake)

                acc,acc5 = attack_acc(fake,label,E)

                
                print("Seed:{} Top1/Top5:{:.3f}/{:.3f}\t".format(random_seed, acc,acc5))
                res.append(acc)
                res5.append(acc5)
                

                if (random_seed+1)%5 == 0:      
                    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
                    std = statistics.stdev(res)
                    std5 = statistics.stdev(res5)

                    print("Top1/Top5:{:.3f}/{:.3f}, std top1/top5:{:.3f}/{:.3f}".format(acc, acc_5, std, std5))

                    aver_acc += acc / N
                    aver_acc5 += acc5 / N
                    aver_std += std / N
                    aver_std5 +=  std5 / N
            print('N_succesful',N_succesful,N_failure)


    return aver_acc, aver_acc5, aver_std, aver_std5



def eval_accuracy(G, E, save_dir, args):
    
    successful_imgs, _ = gen_samples(G, E, save_dir, args.improved_flag)
    
    aver_acc, aver_acc5, \
    aver_std, aver_std5 = accuracy(successful_imgs, E)
    
    
    return aver_acc, aver_acc5, aver_std, aver_std5

def acc_class(filename,fake_dir,E):
    
    E.eval()

    sucessful_fake = np.load(fake_dir + 'success.npy',allow_pickle=True)  
    sucessful_imgs = sucessful_fake.item().get('sucessful_imgs')
    sucessful_label = sucessful_fake.item().get('label')
    sucessful_imgs = concatenate_list(sucessful_imgs)
    sucessful_label = concatenate_list(sucessful_label)

    N_img = 5
    N_id = 300
    with torch.no_grad():
        acc = np.zeros(N_id)
        for id in range(N_id):                
            index = sucessful_label == id
            acc[id] = sum(index)
            
    acc=acc*100.0/N_img 
    print('acc',acc)
    csv_file = '{}acc_class.csv'.format(filename)
    print('csv_file',csv_file)
    import csv
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        for i in range(N_id):
            # writer.writerow(['{}'.format(i),'{}'.format(acc[i])])
            writer.writerow([i,acc[i]])

def eval_acc_class(G, E, save_dir, prefix, args):
    
    successful_imgs, _ = gen_samples(G, E, save_dir, args.improved_flag)
    
    filename = "{}/{}_".format(prefix, args.loss)
    
    acc_class(filename,successful_imgs,E)

