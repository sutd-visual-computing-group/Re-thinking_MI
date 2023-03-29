from argparse import  ArgumentParser
from metrics.KNN_dist import eval_KNN
from metrics.eval_accuracy import eval_accuracy, eval_acc_class
from metrics.fid import eval_fid
from utils import load_json, get_attack_model
import os
import csv 

parser = ArgumentParser(description='Evaluation')
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/ffhq.json')    

args = parser.parse_args()


def init_attack_args(cfg):
    if cfg["attack"]["method"] =='kedmi':
        args.improved_flag = True
        args.clipz = True
        args.num_seeds = 1
    else:
        args.improved_flag = False
        args.clipz = False
        args.num_seeds = 5

    if cfg["attack"]["variant"] == 'L_logit' or cfg["attack"]["variant"] == 'ours':
        args.loss = 'logit_loss'
    else:
        args.loss = 'cel'

    if cfg["attack"]["variant"] == 'L_aug' or cfg["attack"]["variant"] == 'ours':
        args.classid = '0,1,2,3'
    else:
        args.classid = '0'


if __name__ == '__main__':
    # Load Data
    cfg = load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)

    # Save dir
    if args.improved_flag == True:
        prefix = os.path.join(cfg["root_path"], "kedmi_300ids") 
    else:
        prefix = os.path.join(cfg["root_path"], "gmi_300ids") 
    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]), cfg["attack"]["variant"])
    prefix = os.path.join(prefix, save_folder)
    save_dir = os.path.join(prefix, "latent")
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))

    # Load models
    _, E, G, _, _, _, _ = get_attack_model(args, cfg, eval_mode=True)

    # Metrics
    metric = cfg["attack"]["eval_metric"].split(',')
    fid = 0
    aver_acc, aver_acc5, aver_std, aver_std5 = 0, 0, 0, 0
    knn = 0, 0
    nsamples = 0 
    dataset, model_types = '', ''


    
    for metric_ in metric:
        metric_ = metric_.strip()
        if metric_ == 'fid':
            fid, nsamples = eval_fid(G=G, E=E, save_dir=save_dir, cfg=cfg, args=args)
        elif metric_ == 'acc':
            aver_acc, aver_acc5, aver_std, aver_std5 = eval_accuracy(G=G, E=E, save_dir=save_dir, args=args)
        elif metric_ == 'knn':
            knn = eval_KNN(G=G, E=E, save_dir=save_dir, KNN_real_path=cfg["dataset"]["KNN_real_path"], args=args)
       
    csv_file = os.path.join(prefix, 'Eval_results.csv') 
    if not os.path.exists(csv_file):
        header = ['Save_dir', 'Method', 'Succesful_samples',                    
                    'acc','std','acc5','std5',
                    'fid','knn']
        with open(csv_file, 'w') as f:                
            writer = csv.writer(f)
            writer.writerow(header)
    
    fields=['{}'.format(save_dir), 
            '{}'.format(cfg["attack"]["method"]),
            '{}'.format(cfg["attack"]["variant"]),
            '{:.2f}'.format(aver_acc),
            '{:.2f}'.format(aver_std),
            '{:.2f}'.format(aver_acc5),
            '{:.2f}'.format(aver_std5),
            '{:.2f}'.format(fid),
            '{:.2f}'.format(knn)]
    
    print("---------------Evaluation---------------")
    print('Method: {} '.format(cfg["attack"]["method"]))

    print('Variant: {}'.format(cfg["attack"]["variant"]))
    print('Top 1 attack accuracy:{:.2f} +/- {:.2f} '.format(aver_acc, aver_std))
    print('Top 5 attack accuracy:{:.2f} +/- {:.2f} '.format(aver_acc5, aver_std5))
    print('KNN distance: {:.3f}'.format(knn))
    print('FID score: {:.3f}'.format(fid))      
    
    print("----------------------------------------")  
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
