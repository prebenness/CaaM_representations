# train.py
#!/usr/bin/env	python3

import os
import random
#debug
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import sys
import argparse
import time
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)
from conf import settings, global_config as cfg
from utils import get_network, get_dataloader, get_val_dataloader, WarmUpLR, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, \
    update, get_mean_std, Acc_Per_Context, Acc_Per_Context_Class, penalty, cal_acc, get_custom_network, get_custom_network_vit, \
    save_model, load_model, get_parameter_number, init_training_dataloader
from train_module import train_env_ours, auto_split, refine_split, update_pre_optimizer, update_bias_optimizer, auto_cluster
from eval_module import eval_training, eval_best, eval_mode, evaluate_rebias
from timm.scheduler import create_scheduler
from dataset_utils import get_generic_dataloader, get_imagenet_dataloader_env, get_pre_dataloader, get_bias_dataloader



def train(epoch):

    start = time.time()
    net.train()
    train_correct = 0.
    num_updates = epoch * len(train_loader)

    # for batch_index, (images, labels) in enumerate(train_loader):
    for batch_index, (images, labels, _) in enumerate(train_loader):
        if 't2tvit' in args.net and training_opt['optim']['sched']=='cosine':
            lr_scheduler.step_update(num_updates=num_updates)
        else:
            if epoch <= training_opt['warm']:
                warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_correct, train_acc = cal_acc(outputs, labels)
        train_correct += batch_correct

        num_updates += 1

        if batch_index % training_opt['print_batch'] == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tAcc: {:0.4f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                train_acc,
                epoch=epoch,
                trained_samples=batch_index * training_opt['batch_size'] + len(images),
                total_samples=len(train_loader.dataset)
            ))

    finish = time.time()
    train_acc_all = train_correct / len(train_loader.dataset)

    print('epoch {} training time consumed: {:.2f}s \t Train Acc: {:.4f}'.format(epoch, finish - start, train_acc_all))
    return train_acc_all



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='load the config file')
    parser.add_argument('--net', type=str, default='resnet', help='net type')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('--multigpu', action='store_true', default=False, help='use multigpu or not')
    parser.add_argument('--name', type=str, default=None, help='experiment name')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--eval', type=str, default=None, help='the model want to eval')
    parser.add_argument('--local_rank', type=int, default=0, help='the model want to eval')
    parser.add_argument('--worker', default=16, type=int, help='num of worker')
    args = parser.parse_args()

    # ============================================================================
    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    args.net = config['net']
    # args.debug = False
    training_opt = config['training_opt']
    variance_opt = config['variance_opt']
    exp_name = args.name or config['exp_name']

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join('results', f'{timestamp}_{exp_name}')
    os.makedirs(out_dir, exist_ok=True)
    cfg.out_dir = out_dir

    cfg.network_type = config['dataset']
    if config['dataset'] == 'MNIST':
        cfg.mean, cfg.std = settings.MNIST_TRAIN_MEAN, settings.MNIST_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 28, 28), 28          # For simplicity just repeat first channel
        cfg.num_classes, cfg.num_samples = 10, 60_000
    elif config['dataset'] == 'CIFAR10':
        cfg.mean, cfg.std = settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 32, 32), 32
        cfg.num_classes, cfg.num_samples = 10, 60_000
    elif config['dataset'] == 'CIFAR100':
        cfg.mean, cfg.std = settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 32, 32), 32
        cfg.num_classes, cfg.num_samples = 100, 60_000
    else:
        raise ValueError(f'Dataset {config["dataset"]} is not supported')

    if 'mixup' in training_opt and training_opt['mixup'] == True:
        print('use mixup ...')
    # ============================================================================
    # SEED
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(training_opt['seed'])
    if if_cuda:
        torch.cuda.manual_seed(training_opt['seed'])
        torch.cuda.manual_seed_all(training_opt['seed'])
    random.seed(training_opt['seed'])
    np.random.seed(training_opt['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ============================================================================
    # MODEL
    if variance_opt['mode'] in ['ours']:
        if config['net'] == 'vit':
            net = get_custom_network_vit(args, variance_opt)
        else:
            net = get_custom_network(args, variance_opt)
    else:
        net = get_network(args)
    if 'env_type' in variance_opt and variance_opt['env_type'] in ['auto-baseline', 'auto-iter'] and variance_opt['from_scratch']:
        print('load reference model ...')
        ref_arg = argparse.ArgumentParser()
        ref_arg.net = variance_opt['ref_model']
        ref_arg.gpu = args.gpu
        ref_arg.multigpu = args.multigpu
        ref_net = get_network(ref_arg)
        load_model(ref_net, variance_opt['ref_model_path'])
        ref_net = nn.DataParallel(ref_net)
        ref_net.eval()
        print('Done.')
    get_parameter_number(net)


    # ============================================================================
    # DATA PREPROCESSING

    train_loader_init, val_loader_init = get_generic_dataloader(
        config['train_root'], batch_size=training_opt['batch_size'],
        train=True, val_data='NOT_IMAGENET'
    )

    if variance_opt.get('env_type') in ['auto-baseline', 'auto-iter', 'auto-iter-cluster']:
        if variance_opt['env_type'] == 'auto-iter-cluster':
            pre_train_loader, _, __ = train_loader_init.get_pre_dataloader(batch_size=128, num_workers=4, shuffle=False, n_env=variance_opt['n_env'])
        else:
            pre_train_loader, pre_optimizer, pre_schedule = get_pre_dataloader(
                config, config['train_root'], batch_size=512, train=True, num_workers=args.worker,
                n_env=variance_opt['n_env']
            )
        if variance_opt['from_scratch']:
            pre_split_softmax, pre_split = auto_split(ref_net, pre_train_loader, pre_optimizer, pre_schedule, pre_train_loader.dataset.soft_split)
            np.save('misc/unbalance_'+exp_name+'.npy', pre_split.detach().cpu().numpy())
            pre_train_loader.dataset.soft_split = pre_split
            exit()
        else:
            ## Make random pre-split
            #pre_split = np.load('misc/unbalance_bagnet18_imagenet9_presplit.npy')
            #pre_split = torch.from_numpy(pre_split).cuda()
            pre_split = torch.randn((cfg.num_samples, cfg.num_classes))
            pre_train_loader.dataset.soft_split = torch.nn.Parameter(torch.randn_like(pre_split))
            pre_split_softmax = F.softmax(pre_split, dim=-1)

        pre_split = torch.zeros_like(pre_split_softmax).scatter_(1, torch.argmax(pre_split_softmax, 1).unsqueeze(1), 1)
    else:
        pre_split = None

    if 'resnet' in args.net:
        dim_classifier = 512
    else:
        dim_classifier = 256
    if 'env_type' in variance_opt and variance_opt['env_type'] == 'auto-iter':
        bias_classifier = nn.Linear(dim_classifier, training_opt['classes']).cuda()
        bias_optimizer, bias_schedule = update_bias_optimizer(bias_classifier.parameters())
        bias_dataloader = get_bias_dataloader(config['train_root'], batch_size=512, num_workers=args.worker)

    if 'env' in variance_opt:
        train_loader = get_imagenet_dataloader_env(config, config['train_root'], batch_size=training_opt['batch_size'],
                                                   train=True, num_workers=args.worker, pre_split=pre_split)
    else:
        train_loader = train_loader_init


    val_loader = get_generic_dataloader(
        config['val_root'], batch_size=training_opt['val_batch_size'],
        train=False, val_data='NOT_IMAGENET',
    )
    test_loader = get_generic_dataloader(
        config['test_root'], batch_size=training_opt['val_batch_size'],
        train=False, val_data='NOT_IMAGENET',
    )

    loss_function = nn.CrossEntropyLoss()

    if variance_opt['mode'] in ['ours']:
        assert isinstance(net, list)
        if variance_opt['sp_flag']:
            optimizer = []
            ### add classifier optimizer
            optimizer.append(optim.SGD(nn.ModuleList(net[:-1]).parameters(), lr=training_opt['lr'], momentum=0.9, weight_decay=5e-4))
            optimizer.append(optim.SGD(net[-1].parameters(), lr=training_opt['lr']*1.0, momentum=0.9, weight_decay=5e-4))
            train_scheduler = [optim.lr_scheduler.MultiStepLR(optimizer_, milestones=training_opt['milestones'], gamma=0.2) for optimizer_ in optimizer]
            iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
            warmup_scheduler = [WarmUpLR(optimizer_, iter_per_epoch * training_opt['warm']) for optimizer_ in optimizer]
        else:
            optimizer = []
            if 't2tvit' in args.net:
                optimizer.append(optim.AdamW(nn.ModuleList(net).parameters(), lr=training_opt['lr'], weight_decay=0.03))
                if training_opt['optim']['sched'] == 'cosine':
                    lr_scheduler, num_epochs = create_scheduler(argparse.Namespace(**training_opt['optim']), optimizer[0])
                    start_epoch = 0
                    lr_scheduler.step(start_epoch)
                else:
                    train_scheduler = [optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=training_opt['milestones'], gamma=0.2)]  # learning rate decay
                    iter_per_epoch = len(train_loader[0])
                    warmup_scheduler = [WarmUpLR(optimizer[0], iter_per_epoch * training_opt['warm'])]
            else:
                optimizer.append(optim.SGD(nn.ModuleList(net).parameters(), lr=training_opt['lr'], momentum=0.9, weight_decay=5e-4))
                train_scheduler = [optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=training_opt['milestones'], gamma=0.2)]  # learning rate decay
                iter_per_epoch = len(train_loader[0])
                warmup_scheduler = [WarmUpLR(optimizer[0], iter_per_epoch * training_opt['warm'])]

    else:
        if 't2tvit' in args.net:
            optimizer = optim.AdamW(net.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            if training_opt['optim']['sched']=='cosine':
                lr_scheduler, num_epochs = create_scheduler(argparse.Namespace(**training_opt['optim']), optimizer)
                start_epoch = 0
                lr_scheduler.step(start_epoch)
            else:
                train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=0.2)  # learning rate decay
                iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
                warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * training_opt['warm'])
        else:
            optimizer = optim.SGD(net.parameters(), lr=training_opt['lr'], momentum=0.9, weight_decay=5e-4)
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=0.2)  # learning rate decay
            iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * training_opt['warm'])


    if config['resume']:
        recent_folder = most_recent_folder(os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net, exp_name)

    if args.debug:
        checkpoint_path = os.path.join(checkpoint_path, 'debug')

    #use tensorboard
    log_dir = os.path.join(cfg.out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, exp_name))


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_epoch = 0
    best_val_acc = 0.0
    best_train_acc = 0.0

    if 'pretrain' in config and config['pretrain'] is not None:
        state_dict = torch.load(config['pretrain'])
        net.load_state_dict(state_dict, strict=False)
        print('Loaded pretrained model...')
    if config['resume']:
        best_weights = best_acc_weights(os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(os.path.join(cfg.out_dir, 'checkpoints'), args.net, recent_folder))

    if args.eval is not None:
        scores = eval_mode(config, net, val_loader, args.eval)
        print(scores)
        exit()

    num_epochs = 1 if args.debug else training_opt['epoch']
    for epoch in range(1, num_epochs + 1):
        if 't2tvit' in args.net and training_opt['optim']['sched']=='cosine':
            lr_scheduler.step(epoch)
        else:
            if epoch > training_opt['warm']:
                if isinstance(train_scheduler, list):
                    for train_scheduler_ in train_scheduler:
                        train_scheduler_.step()
                else:
                    train_scheduler.step()
        if config['resume']:
            if epoch <= resume_epoch:
                continue

        ### update split
        if variance_opt['env_type'] == 'auto-iter' and epoch>=variance_opt['split_renew'] and (epoch-variance_opt['split_renew'])%variance_opt['split_renew_iters']==0:
            bias_classifier, bias_optimizer = refine_split(bias_optimizer, bias_schedule, bias_classifier, bias_dataloader, net[-1])
            pre_optimizer, pre_schedule = update_pre_optimizer(pre_train_loader.dataset.soft_split)
            updated_split_softmax, updated_split = auto_split([net[-1], bias_classifier], pre_train_loader, pre_optimizer, pre_schedule, pre_train_loader.dataset.soft_split)
            # pre_train_loader.dataset.soft_split = torch.nn.Parameter(updated_split)
            pre_train_loader.dataset.soft_split = torch.nn.Parameter(torch.randn_like(updated_split))

            # updata dataloader
            updated_split_onehot = torch.zeros_like(updated_split_softmax).scatter_(1, torch.argmax(updated_split_softmax, 1).unsqueeze(1), 1)
            train_loader = get_imagenet_dataloader_env(config, config['train_root'], batch_size=training_opt['batch_size'],
                                                       train=True, num_workers=args.worker, pre_split=updated_split_onehot)

            bias_classifier = nn.Linear(dim_classifier, training_opt['classes']).cuda()
            bias_optimizer, bias_schedule = update_bias_optimizer(bias_classifier.parameters())
            print('Update Dataloader Done')


        elif variance_opt['env_type'] == 'auto-iter-cluster' and epoch>=variance_opt['split_renew'] and (epoch-variance_opt['split_renew'])%variance_opt['split_renew_iters']==0:
            updated_split = auto_cluster(pre_train_loader, net[-1], training_opt, variance_opt)
            # updata dataloader
            train_loader = train_loader_init.get_env_dataloader(config, training_opt['batch_size'], num_workers=4, shuffle=True, pre_split=updated_split)
            print('Update Dataloader Done')


        if 'env' in variance_opt:
            if variance_opt['mode'] == 'ours':
                train_acc = train_env_ours(epoch, net, train_loader, args, training_opt, variance_opt, loss_function, optimizer, warmup_scheduler)
        else:
            train_acc = train(epoch)

        val_acc = evaluate_rebias(val_loader, net, config, num_classes=cfg.num_classes, key='biased')['f_acc']
        if val_acc > best_val_acc:
            # torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            save_model(net, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = val_acc
            best_epoch = epoch
            best_train_acc  = train_acc

        if not epoch % training_opt['save_epoch']:
            save_model(net, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        print("Best Acc: %.4f \t Train Acc: %.4f \t Best Epoch: %d" %(best_acc, best_train_acc, best_epoch))


    print('Evaluate Best Epoch %d on test set ...' %(best_epoch))
    scores = eval_best(config, args, net, test_loader, loss_function, checkpoint_path, best_epoch)
    print(scores)
    txt_out_dir = os.path.join(cfg.out_dir, 'results')
    os.makedirs(txt_out_dir, exist_ok=True)
    txt_write = open(os.path.join(txt_out_dir, f'{exp_name}.txt'), 'w')
    txt_write.write(f'Best train acc: {best_train_acc}\n')
    txt_write.write(f'Best val acc: {best_acc}\n')
    txt_write.write(f'Test scores: {scores}\n')
    txt_write.close()
    writer.close()
