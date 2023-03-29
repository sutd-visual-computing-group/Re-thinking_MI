import torch, os, engine, utils
import torch.nn as nn
from argparse import  ArgumentParser
from models import classify


parser = ArgumentParser(description='Train Classifier')
parser.add_argument('--configs', type=str, default='./config/celeba/training_classifiers/classify.json')  

args = parser.parse_args()



def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    mode = args["dataset"]["mode"]

    resume_path = args[args["dataset"]["model_name"]]["resume"]
    net = classify.get_classifier(model_name=model_name, mode=mode, n_classes=n_classes, resume_path=resume_path)
    
    print(net)

    optimizer = torch.optim.SGD(params=net.parameters(),
							    lr=args[model_name]['lr'], 
            					momentum=args[model_name]['momentum'], 
            					weight_decay=args[model_name]['weight_decay'])
	
    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).to(args['dataset']['device'])

    mode = args["dataset"]["mode"]
    n_epochs = args[model_name]['epochs']
    print("Start Training!")
	
    if mode == "reg":
        best_model, best_acc = engine.train_reg(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
    elif mode == "vib":
        best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
	
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_{:.2f}_allclass.tar").format(model_name, best_acc[0]))


if __name__ == '__main__':

    cfg = utils.load_json(json_file=args.configs)

    root_path = cfg["root_path"]
    log_path = os.path.join(root_path, "target_logs")
    model_path = os.path.join(root_path, "target_ckp")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)


    model_name = cfg['dataset']['model_name']
    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    print("TRAINING %s" % model_name)
    utils.print_params(cfg["dataset"], cfg[model_name], dataset=cfg['dataset']['name'])

    train_file = cfg['dataset']['train_file_path']
    test_file = cfg['dataset']['test_file_path']
    _, trainloader = utils.init_dataloader(cfg, train_file, mode="train")
    _, testloader = utils.init_dataloader(cfg, test_file, mode="test")

    main(cfg, model_name, trainloader, testloader)
