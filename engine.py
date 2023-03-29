import torch, os, time, utils
import torch.nn as nn
from copy import deepcopy
from utils import *
from models.discri import MinibatchDiscriminator, DGWGAN
from models.generator import Generator
from models.classify import *
from tensorboardX import SummaryWriter


def test(model, criterion=None, dataloader=None, device='cuda'):
    tf = time.time()
    model.eval()
    loss, cnt, ACC, correct_top5 = 0.0, 0, 0,0
    with torch.no_grad():
        for i,(img, iden) in enumerate(dataloader):
            img, iden = img.to(device), iden.to(device)

            bs = img.size(0)
            iden = iden.view(-1)
            _,out_prob = model(img)
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()


            _, top5 = torch.topk(out_prob,5, dim = 1)  
            for ind,top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    correct_top5 += 1
        
            cnt += bs

    return ACC*100.0/cnt, correct_top5*100.0/cnt

def train_reg(args, model, criterion, optimizer, trainloader, testloader, n_epochs, device='cuda'):
    best_ACC = (0.0, 0.0)
        
    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()
		
        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, criterion, testloader)

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc[0]))

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC

def train_vib(args, model, criterion, optimizer, trainloader, testloader, n_epochs, device='cuda'):
	best_ACC = (0.0, 0.0)
	
	for epoch in range(n_epochs):
		tf = time.time()
		ACC, cnt, loss_tot = 0, 0, 0.0
		
		for i, (img, iden) in enumerate(trainloader):
			img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
			bs = img.size(0)
			iden = iden.view(-1)
			
			___, out_prob, mu, std = model(img, "train")
			cross_loss = criterion(out_prob, one_hot)
			info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
			loss = cross_loss + beta * info_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			out_iden = torch.argmax(out_prob, dim=1).view(-1)
			ACC += torch.sum(iden == out_iden).item()
			loss_tot += loss.item() * bs
			cnt += bs

		train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
		test_loss, test_acc = test(model, criterion, testloader)

		interval = time.time() - tf
		if test_acc[0] > best_ACC[0]:
			best_ACC = test_acc
			best_model = deepcopy(model)
			
		print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc[0]))
		

	print("Best Acc:{:.2f}".format(best_ACC[0]))
	return best_model, best_ACC


def get_T(model_name_T, cfg):
    if model_name_T.startswith("VGG16"):
        T = VGG16(cfg['dataset']["n_classes"])
    elif model_name_T.startswith('IR152'):
        T = IR152(cfg['dataset']["n_classes"])
    elif model_name_T == "FaceNet64":
        T = FaceNet64(cfg['dataset']["n_classes"])
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(cfg[cfg['dataset']['model_name']]['cls_ckpts'])
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    return T


def train_specific_gan(cfg):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name_T = cfg['dataset']['model_name']
    batch_size = cfg[model_name_T]['batch_size']
    z_dim = cfg[model_name_T]['z_dim']
    n_critic = cfg[model_name_T]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Create save folders
    root_path = cfg["root_path"]
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, model_name_T))
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)


    # Log file
    log_path = os.path.join(save_model_dir, "attack_logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = "improvedGAN_{}.txt".format(model_name_T)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    writer = SummaryWriter(log_path)


    # Load target model
    T = get_T(model_name_T=model_name_T, cfg=cfg)

    # Dataset
    dataset, dataloader = utils.init_dataloader(cfg, file_path, cfg[model_name_T]['batch_size'], mode="gan")

    # Start Training
    print("Training GAN for %s" % model_name_T)
    utils.print_params(cfg["dataset"], cfg[model_name_T])

    G = Generator(cfg[model_name_T]['z_dim'])
    DG = MinibatchDiscriminator()
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))

    entropy = HLoss()

    step = 0
    for epoch in range(cfg[model_name_T]['epochs']):
        start = time.time()
        _, unlabel_loader1 = init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)

        for i, imgs in enumerate(dataloader):
            current_iter = epoch * len(dataloader) + i + 1

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            x_unlabel = unlabel_loader1.next()
            x_unlabel2 = unlabel_loader2.next()
            
            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            y_prob = T(imgs)[-1]
            y = torch.argmax(y_prob, dim=1).view(-1)
            

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake =  DG(f_imgs)

            loss_lab = softXEnt(output_label, y_prob)
            loss_unlab = 0.5*(torch.mean(F.softplus(log_sum_exp(output_unlabel)))-torch.mean(log_sum_exp(output_unlabel))+torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab
            
            acc = torch.mean((output_label.max(1)[1] == y).float())
            
            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            writer.add_scalar('loss_label_batch', loss_lab, current_iter)
            writer.add_scalar('loss_unlabel_batch', loss_unlab, current_iter)
            writer.add_scalar('DG_loss_batch', dg_loss, current_iter)
            writer.add_scalar('Acc_batch', acc, current_iter)

            # train G
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                mom_unlabel, _ = DG(x_unlabel2)

                mom_gen = torch.mean(mom_gen, dim = 0)
                mom_unlabel = torch.mean(mom_unlabel, dim = 0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss  

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                writer.add_scalar('G_loss_batch', g_loss, current_iter)

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))

        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "improved_{}_G.tar".format(dataset_name)))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "improved_{}_D.tar".format(dataset_name)))

        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "improved_celeba_img_{}.png".format(epoch)), nrow = 8)


def train_general_gan(cfg):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name = cfg['dataset']['model_name']
    lr = cfg[model_name]['lr']
    batch_size = cfg[model_name]['batch_size']
    z_dim = cfg[model_name]['z_dim']
    epochs = cfg[model_name]['epochs']
    n_critic = cfg[model_name]['n_critic']
    dataset_name = cfg['dataset']['name']


    # Create save folders
    root_path = cfg["root_path"]
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, 'general_GAN'))
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)


    # Log file
    log_path = os.path.join(save_model_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = "GAN_{}.txt".format(dataset_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    writer = SummaryWriter(log_path)


    # Dataset
    dataset, dataloader = init_dataloader(cfg, file_path, batch_size, mode="gan")


    # Start Training
    print("Training general GAN for %s" % dataset_name)
    utils.print_params(cfg["dataset"], cfg[model_name])

    G = Generator(z_dim)
    DG = DGWGAN(3)
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):
            
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            
            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)
            
            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data, DG=DG)
            dg_loss = - wd + gp * 10.0
            
            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G

            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))
        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch)), nrow = 8)
        
        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "celeba_G.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "celeba_D.tar"))

def train_augmodel(cfg):
    # Hyperparams
    target_model_name = cfg['train']['target_model_name']
    student_model_name = cfg['train']['student_model_name']
    device = cfg['train']['device']
    lr = cfg['train']['lr']
    temperature = cfg['train']['temperature']
    dataset_name = cfg['dataset']['name']
    n_classes = cfg['dataset']['n_classes']
    batch_size = cfg['dataset']['batch_size']
    seed = cfg['train']['seed']
    epochs = cfg['train']['epochs']
    log_interval = cfg['train']['log_interval']

    
    # Create save folder
    save_dir = os.path.join(cfg['root_path'], dataset_name)
    save_dir = os.path.join(save_dir, '{}_{}_{}_{}'.format(target_model_name, student_model_name, lr, temperature))
    os.makedirs(save_dir, exist_ok=True)

    # Log file    
    now = datetime.now() # current date and time
    log_file = "studentKD_logs_{}.txt".format(now.strftime("%m_%d_%Y_%H_%M_%S"))
    utils.Tee(os.path.join(save_dir, log_file), 'w')
    torch.manual_seed(seed)


    kwargs = {'batch_size': batch_size}
    if device == 'cuda':
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    
    # Get models
    teacher_model = get_augmodel(target_model_name, n_classes, cfg['train']['target_model_ckpt'])
    model = get_augmodel(student_model_name, n_classes)
    model = model.to(device)
    print('Target model {}: {} params'.format(target_model_name, count_parameters(model)))
    print('Augmented model {}: {} params'.format(student_model_name, count_parameters(teacher_model)))


    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    

    # Get dataset
    _, dataloader_train = init_dataloader(cfg, cfg['dataset']['gan_file_path'], batch_size, mode="gan")
    _, dataloader_test = init_dataloader(cfg, cfg['dataset']['test_file_path'], batch_size, mode="test")


    # Start training
    top1,top5 = test(teacher_model, dataloader=dataloader_test)
    print("Target model {}: top 1 = {}, top 5 = {}".format(target_model_name, top1, top5))


    loss_function = nn.KLDivLoss(reduction='sum')
    teacher_model.eval()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, data in enumerate(dataloader_train):
            data  = data.to(device)

            curr_batch_size = len(data)
            optimizer.zero_grad()
            _, output_t = teacher_model(data)
            _, output = model(data)

            loss = loss_function(
                F.log_softmax(output / temperature, dim=-1),
                F.softmax(output_t / temperature, dim=-1)
            ) / (temperature * temperature) / curr_batch_size


            loss.backward()
            optimizer.step()
            
            if (log_interval > 0) and (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader_train.dataset),
                    100. * batch_idx / len(dataloader_train), loss.item()))
                  
        scheduler.step()
        top1, top5 = test(model, dataloader=dataloader_test)
        print("epoch {}: top 1 = {}, top 5 = {}".format(epoch, top1, top5))
        
        if (epoch+1)%10 == 0:
            save_path = os.path.join(save_dir, "{}_{}_kd_{}_{}.pt".format(target_model_name, student_model_name, seed, epoch+1))
            torch.save({'state_dict':model.state_dict()}, save_path)
