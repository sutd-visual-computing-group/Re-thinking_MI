# Implementation of paper "Re-thinking Model Inversion Attacks Against Deep Neural Networks" - CVPR 2023
[Paper](https://arxiv.org/pdf/2304.01669.pdf) | [Project page](https://ngoc-nguyen-0.github.io/re-thinking_model_inversion_attacks/)
## 1. Setup Environment
This code has been tested with Python 3.7, PyTorch 1.11.0 and Cuda 11.3. 

```
conda create -n MI python=3.7

conda activate MI

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

## 2. Prepare Dataset & Checkpoints

* Dowload CelebA and FFHQ dataset at the official website.
- CelebA: download and extract the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Then, place the `img_align_celeba` folder to `.\datasets\celeba`

- FFHQ: download and extract the [FFHQ](https://github.com/NVlabs/ffhq-dataset). Then, place the `thumbnails128x128` folder to `.\datasets\ffhq`

* Download meta data for the experiments at: https://drive.google.com/drive/folders/1kq4ArFiPmCWYKY7iiV0WxxUSXtP70bFQ?usp=sharing


* We use the same target models and GAN as previous papers. You can download target models and generator at https://drive.google.com/drive/folders/1kq4ArFiPmCWYKY7iiV0WxxUSXtP70bFQ?usp=sharing

Otherwise, you can train the target classifier and GAN as follow:
  

### 2.1. Training the target classifier (Optional)

- Modify the configuration in `.\config\celeba\classify.json`
- Then, run the following command line to get the target model
  ```
  python train_classifier.py
  ```

### 2.2. Training GAN (Optional)

SOTA MI attacks work with a general GAN[1]. However, Inversion-Specific GANs[2] help improve the attack accuracy. In this repo, we provide codes for both training general GAN and Inversion-Specific GAN.

#### 2.2.1. Build a inversion-specific GAN 
* Modify the configuration in
  * `./config/celeba/training_GAN/specific_gan/celeba.json` if training a Inversion-Specific GAN on CelebA (KEDMI[2]).
  * `./config/celeba/training_GAN/specific_gan/ffhq.json` if training a Inversion-Specific GAN on FFHQ (KEDMI[2]).
  
* Then, run the following command line to get the Inversion-Specific GAN
    ```
    python train_gan.py --configs path/to/config.json --mode "specific"
    ```

#### 2.2.2. Build a general GAN 
* Modify the configuration in
  * `./config/celeba/training_GAN/general_gan/celeba.json` if training a general GAN on CelebA (GMI[1]).
  * `./config/celeba/training_GAN/general_gan/ffhq.json` if training a general GAN on FFHQ (GMI[1]).
  
* Then, run the following command line to get the General GAN
    ```
    python train_gan.py --configs path/to/config.json --mode "general"
    ```

## 3. Learn augmented models
We provide code to train augmented models (i.e., `efficientnet_b0`, `efficientnet_b1`, and `efficientnet_b2`) from a ***target model***.
* Modify the configuration in
  * `./config/celeba/training_augmodel/celeba.json` if training an augmented model on CelebA
  * `./config/celeba/training_augmodel/ffhq.json` if training an augmented model on FFHQ
  
* Then, run the following command line to train augmented models
    ```
    python train_augmented_model.py --configs path/to/config.json
    ```

Pretrained augmented models and p_reg can be downloaded at https://drive.google.com/drive/u/2/folders/1kq4ArFiPmCWYKY7iiV0WxxUSXtP70bFQ

***We remark that if you train augmented models, please do not use our p_reg***. Delete files in `./p_reg/` before inversion. Our code will automatically estimate p_reg with new augmented models.

## 4. Model Inversion Attack

* Modify the configuration in
  * `./config/celeba/attacking/celeba.json` if training an augmented model on CelebA
  * `./config/celeba/attacking/ffhq.json` if training an augmented model on FFHQ

* Important arguments:
  * `method`: select the method either ***gmi*** or ***kedmi***
  * `variant` select the variant either ***baseline***, ***L_aug***, ***L_logit***, or ***ours***

* Then, run the following command line to attack
    ```
    python recovery.py --configs path/to/config.json
    ```

## 5. Evaluation

After attack, use the same configuration file to run the following command line to get the result:\
```
python evaluation.py --configs path/to/config.json
```

## Acknowledgements
We gratefully acknowledge the following works:
- Knowledge-Enriched-Distributional-Model-Inversion-Attacks: https://github.com/SCccc21/Knowledge-Enriched-DMI
- EfficientNet (Pytorch): https://pytorch.org/vision/stable/models/efficientnet.html
- Experiment Tracking with Weights and Biases : https://www.wandb.com/


## Reference
<a id="1">[1]</a> 
Zhang, Yuheng, et al. "The secret revealer: Generative model-inversion attacks against deep neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.


<a id="2">[2]</a>  Si Chen, Mostafa Kahla, Ruoxi Jia, and Guo-Jun Qi. Knowledge-enriched distributional model inversion attacks. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16178–16187, 2021
