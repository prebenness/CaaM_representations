import os
from collections import OrderedDict

import torch
import numpy as np
import yaml
from progress.bar import Bar

from utils import get_test_dataloader, init_training_dataloader
from models.t2tvit_ours import T2t_vit_7_feature, classifier


def get_model_list(variance_opt):
    num_env = variance_opt['n_env']
    
    model_list = []
    for e in range(num_env):
        if (e <= num_env):
            model_list.append(classifier(num_classes=10, bias=False))

    model_list.append(classifier(num_classes=10))
    model_list.append(T2t_vit_7_feature(num_classes=10, final_k=variance_opt['final_k']))

    # Use GPU
    model_list = [ model_list_.cuda() for model_list_ in model_list ]

    return model_list


def update_key_names(d):
    new_d = OrderedDict()
    for old_key in d:
        new_key = '.'.join(old_key.split('.')[1:])
        new_d[new_key] = d[old_key]

    return new_d


def load_pretrained(config):
    # Fake args
    trained_model_path = os.path.join('pretrain_model', 'nico_t2tvit7_ours_caam-best.pth')

    nets = get_model_list(config['variance_opt'])
    state_dict_dict = torch.load(trained_model_path)

    for idx, net in enumerate(nets):
        updated_state_dict = update_key_names(state_dict_dict[idx])
        net.load_state_dict(updated_state_dict)

    print('Loaded pretrained model...')

    return nets


if __name__ == '__main__':
    # Load config
    conf_path = os.path.join('conf', 't2tvit_best_representations.yaml')
    with open(conf_path) as f:
        config = yaml.safe_load(f)
    
    # Load pretrained model
    nets = load_pretrained(config)
    model = nets[-1]

    # Use GPU and set to train mode to False
    model.cuda()
    model.eval()

    # Load dataset
    ## Test split
    mean, std = config['training_opt']['mean'], config['training_opt']['std']
    
    print('Loading test split')
    test_loader = get_test_dataloader(config=config, mean=mean, std=std, num_workers=8, batch_size=128)

    ## Train split
    print('Loading train split')
    train_loader_init = init_training_dataloader(config=config, mean=mean, std=std)
    train_loader, _, _ = train_loader_init.get_pre_dataloader(batch_size=128, num_workers=8, shuffle=False, n_env=config['variance_opt']['n_env'])

    # Forward pass over all data to compute representations
    print('Starting representations computation')
    split_name2loader = { 'train': train_loader, 'test':test_loader }

    for split_name, loader in split_name2loader.items():
        images, style, content = [], [], []
        
        print(f'Split: {split_name}')
        bar = Bar('Processing', max=len(loader), index=0)
        for (x, _, _) in loader:
            x = x.cuda()

            # Estimate causal and spurious signals
            x_causal, x_spurious, x_mix = model(x)

            # Store representations
            images.append(x)
            content.append(x_causal)
            style.append(x_spurious)

            bar.next()

        ## Save images, contents and styles for test and train sets
        images = torch.cat(images, 0).detach().cpu().numpy()
        style = torch.cat(style, 0).detach().cpu().numpy()
        content = torch.cat(content, 0).detach().cpu().numpy()

        outpath = os.path.join('output', 'representations', f'{config["dataset"]}-{config["net"]}')
        os.makedirs(outpath, exist_ok=True)
        np.savez(os.path.join(outpath, f'content_{split_name}.npz'), content)
        np.savez(os.path.join(outpath, f'images_{split_name}.npz'), images)
        np.savez(os.path.join(outpath, f'style_{split_name}.npz'), style)

    print('Representations computed and stored')