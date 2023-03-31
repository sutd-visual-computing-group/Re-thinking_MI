from tqdm import tqdm
import torch
import numpy as np
from scipy import linalg
from metrics import metric_utils
import utils
from attack import reparameterize
import os
from utils import save_tensor_images


device = 'cuda' if torch.cuda.is_available() else 'cpu'
_feature_detector_cache = None
def get_feature_detector():
    global _feature_detector_cache
    if _feature_detector_cache is None:
        _feature_detector_cache = metric_utils.get_feature_detector(
            'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/'
            'metrics/inception-2015-12-05.pt', device)

    return _feature_detector_cache


def postprocess(x):
    """."""
    return ((x * .5 + .5) * 255).to(torch.uint8)


def run_fid(x1, x2):
    # Extract features
    x1 = run_batch_extract(x1, device)
    x2 = run_batch_extract(x2, device)

    npx1 = x1.detach().cpu().numpy()
    npx2 = x2.detach().cpu().numpy()
    mu1 = np.mean(npx1, axis=0)
    sigma1 = np.cov(npx1, rowvar=False)
    mu2 = np.mean(npx2, axis=0)
    sigma2 = np.cov(npx2, rowvar=False)
    frechet = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return frechet


def run_feature_extractor(x):
    assert x.dtype == torch.uint8
    assert x.min() >= 0
    assert x.max() <= 255
    assert len(x.shape) == 4
    assert x.shape[1] == 3
    feature_extractor = get_feature_detector()
    return feature_extractor(x, return_features=True)


def run_batch_extract(x, device, bs=500):
    z = []
    with torch.no_grad():
        for start in tqdm(range(0, len(x), bs), desc='run_batch_extract'):
            stop = start + bs
            x_ = x[start:stop].to(device)
            z_ = run_feature_extractor(postprocess(x_)).cpu()
            z.append(z_)
    z = torch.cat(z)
    return z


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, return_details=False):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    if not return_details:
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    else:
        t1 = diff.dot(diff)
        t2 = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return (t1 + t2), t1, t2


def get_z(improved_gan, save_dir, loop, i, j):
    if improved_gan==True: #KEDMI
        outputs_z = os.path.join(save_dir, "{}_{}_iter_0_{}_dis.npy".format(loop, i, 2399)) 
        outputs_label = os.path.join(save_dir, "{}_{}_iter_0_{}_label.npy".format(loop, i, 2399)) 
        
        dis = np.load(outputs_z, allow_pickle=True)  
        mu = torch.from_numpy(dis.item().get('mu')).to(device)             
        log_var = torch.from_numpy(dis.item().get('log_var')).to(device)
        iden = np.load(outputs_label)
        z = reparameterize(mu, log_var) 
    else: #GMI
        outputs_z = os.path.join(save_dir, "{}_{}_iter_{}_{}_z.npy".format(save_dir, loop, i, j, 2399))
        outputs_label = os.path.join(save_dir, "{}_{}_iter_{}_{}_label.npy".format(save_dir, loop, i, j, 2399)) 
        
        z = np.load(outputs_z)  
        iden = np.load(outputs_label)
        z = torch.from_numpy(z).to(device)
    return z, iden

def gen_samples(G, E, save_dir, improved_gan, n_iden=5, n_img=5):
    total_gen = 0
    seed = 9
    torch.manual_seed(seed)
    img_ids_path = os.path.join(save_dir, 'attack{}_'.format(seed))

    all_sucessful_imgs = []
    all_failure_imgs = []
    
    all_imgs = []                            
    all_fea = []
    all_id = []
    all_sucessful_imgs = []
    all_sucessful_id =[]
    all_sucessful_fea=[]

    all_failure_imgs = []                            
    all_failure_fea = []
    all_failure_id = [] 
    
    E.eval()
    G.eval()
    if not os.path.exists(img_ids_path + 'full.npy'):
        for loop in range(1):
            for i in range(n_iden): #300 ides 
                for j in range(n_img): #5 images/iden
                    z, iden = get_z(improved_gan, save_dir, loop, i, j)
                    z = torch.clamp(z, -1.0, 1.0).float()
                    total_gen = total_gen + z.shape[0]
                    # calculate attack accuracy
                    with torch.no_grad():
                        fake = G(z.to(device))
                        save_tensor_images(fake, os.path.join(save_dir, "gen_{}_{}.png".format(i,j)), nrow = 60)

                        eval_fea, eval_prob = E(utils.low2high(fake))
                        
                        ### successfully attacked samples       
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        sucessful_iden = []
                        failure_iden = []
                        for id in range(iden.shape[0]):
                            if eval_iden[id]==iden[id]:
                                sucessful_iden.append(id)
                            else:
                                failure_iden.append(id)
                        
                        
                        fake = fake.detach().cpu().numpy()
                        eval_fea = eval_fea.detach().cpu().numpy()  
                    
                        all_imgs.append(fake)
                        all_fea.append(eval_fea)
                        all_id.append(iden)
                          
                        if len(sucessful_iden)>0:                              
                            sucessful_iden = np.array(sucessful_iden)                            
                            sucessful_fake = fake[sucessful_iden,:,:,:]                    
                            sucessful_eval_fea = eval_fea[sucessful_iden,:]
                            sucessful_iden = iden[sucessful_iden]
                        else:
                            sucessful_fake = []
                            sucessful_iden = []
                            sucessful_eval_fea = []
                        
                        all_sucessful_imgs.append(sucessful_fake)
                        all_sucessful_id.append(sucessful_iden)
                        all_sucessful_fea.append(sucessful_eval_fea)

                        if len(failure_iden)>0: 
                            failure_iden = np.array(failure_iden)
                            failure_fake = fake[failure_iden,:,:,:]                    
                            failure_eval_fea = eval_fea[failure_iden,:]
                            failure_iden = iden[failure_iden]
                        else:
                            failure_fake = []
                            failure_iden = []
                            failure_eval_fea = []
              
                        all_failure_imgs.append(failure_fake)
                        all_failure_id.append(failure_iden)
                        all_failure_fea.append(failure_eval_fea)     
        np.save(img_ids_path+'full',{'imgs':all_imgs,'label':all_id,'fea':all_fea})
        np.save(img_ids_path+'success',{'sucessful_imgs':all_sucessful_imgs,'label':all_sucessful_id,'sucessful_fea':all_sucessful_fea})
        np.save(img_ids_path+'failure',{'failure_imgs':all_failure_imgs,'label':all_failure_id,'failure_fea':all_failure_fea})
        
    return img_ids_path, total_gen


def concatenate_list(listA):
    result = []
    for i in range(len(listA)):
        val = listA [i]
        if len(val)>0:
            if len(result)==0:
                result = listA[i]
            else:
                result = np.concatenate((result,val))
    return result


def eval_fid(G, E, save_dir, cfg, args):

    successful_imgs,_ = gen_samples(G, E, save_dir, args.improved_flag)
    
    #real data
    target_x = np.load(cfg['dataset']['fid_real_path'])

    # Load Samples
    sucessful_data = np.load(successful_imgs+'success.npy',allow_pickle=True)  
    fake = sucessful_data.item().get('sucessful_imgs')
    
    fake = concatenate_list(fake)

    fake = torch.from_numpy(fake).to(device)
    target_x = torch.from_numpy(target_x).to(device)
    fid = run_fid(target_x, fake)

    return fid, fake.shape[0] 
        