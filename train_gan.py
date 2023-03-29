import engine
from utils import load_json
from argparse import  ArgumentParser


parser = ArgumentParser(description='Train GAN')
parser.add_argument('--configs', type=str, default='./config/celeba/training_GAN/specific_gan/celeba.json') 
parser.add_argument('--mode', type=str, choices=['specific', 'general'], default='specific')
args = parser.parse_args()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
    file = args.configs
    cfg = load_json(json_file=file)

    if args.mode == 'specific':
        engine.train_specific_gan(cfg=cfg)
    elif args.mode == 'general':
        engine.train_general_gan(cfg=cfg)
