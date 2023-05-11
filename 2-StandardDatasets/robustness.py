import os
import argparse

import torch
import torchattacks


from representations import load_pretrained
from conf import settings, global_config as cfg
from dataset_utils import get_generic_dataloader

class CaaMWrapper(torch.nn.Module):
    def __init__(self, nets):
        super().__init__()

        # Set eval mode and GPU
        [ net.cuda() for net in nets ]
        [ net.eval() for net in nets ]

        # Linear classifiers
        self.linears = nets[:-1]
        self.linear_w = torch.stack([net.fc.weight for net in self.linears], 0).mean(0).to(cfg.device)

        # Image embedder
        self.embedder = nets[-1]
    
    def forward(self, x):
        x_causal, _, _ = self.embedder(x)
        y_pred = torch.nn.functional.linear(x_causal, self.linear_w)

        return y_pred

class DummyAttack():
    def __init__(self, m):
        ...

    def set_normalization_used(self, mean, std):
        ...

    def __call__(self, x, y):
        return x


def test_robustness(model_path, attack_factory, test_loader):
    num_corr, num_corr_adv, num_tot = 0, 0, 0
    for (x, y, _) in test_loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        # Load pretrained model
        nets = load_pretrained(model_path)
        model = CaaMWrapper(nets).to(cfg.device)

        # Get clean test preds
        y_pred = model(x)

        # Craft adversarial samples
        attacker = attack_factory(model)
        attacker.set_normalization_used(mean=cfg.mean, std=cfg.std)
        x_adv = attacker(x, y)
        y_pred_adv = model(x_adv)

        num_corr += (y_pred.argmax(dim=1) == y).sum().item()
        num_corr_adv += (y_pred_adv.argmax(dim=1) == y).sum().item()
        num_tot += y.shape[0]

        print(f'Processed {num_tot:5d} of {len(test_loader.dataset):5d} samples, current tally: Clean acc: {num_corr / num_tot:4.3f} Adv acc: {num_corr_adv / num_tot:4.3f}')

    # Compute and store results
    clean_acc = num_corr / num_tot
    adv_acc = num_corr_adv / num_tot

    res_text = f'Test completed: Clean acc: {clean_acc} Adv acc: {adv_acc}'
    print(res_text)

    return clean_acc, adv_acc


def main():
    parser = argparse.ArgumentParser('Compute representations of causal and correlational signals')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'], required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--trained_model', type=str, required=True)
    parser.add_argument('--no_gpu', action='store_true', required=False)


    args = parser.parse_args()

    cfg.device = 'cpu' if args.no_gpu else 'cuda'
    cfg.network_type = args.dataset.upper()
    if args.dataset == 'mnist':
        cfg.mean, cfg.std = settings.MNIST_TRAIN_MEAN, settings.MNIST_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 28, 28), 28          # For simplicity just repeat first channel
        cfg.num_classes = 10
    elif args.dataset == 'cifar10':
        cfg.mean, cfg.std = settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 32, 32), 32
        cfg.num_classes = 10
    elif args.dataset == 'cifar100':
        cfg.mean, cfg.std = settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 32, 32), 32
        cfg.num_classes = 100
    else:
        raise ValueError(f'Dataset {args.dataset} is not supported')


    # Load dataset
    test_loader = get_generic_dataloader(
        os.path.join(args.data_root, 'test'), batch_size=128,
        train=False, val_data='NOT_IMAGENET'
    )


    attack_factories = {
        'dummy_attacker': DummyAttack,
        'pgd20_linf': lambda m: torchattacks.PGD(m, eps=8/255, alpha=2/255, steps=20, random_start=True),
        'pgd40_linf': lambda m: torchattacks.PGD(m, eps=8/255, alpha=4/255, steps=40, random_start=True),
        'pgd20_l2': lambda m: torchattacks.PGDL2(m, eps=1.0, alpha=0.2, steps=20, random_start=True),
        'pgd40_l2': lambda m: torchattacks.PGDL2(m, eps=1.0, alpha=0.2, steps=40, random_start=True),
        'fgsm_linf': lambda m: torchattacks.FGSM(m, eps=8/255),
        'cw20_l2': lambda m: torchattacks.CW(m, c=1, kappa=0, steps=20),
        'cw40_l2': lambda m: torchattacks.CW(m, c=1, kappa=0, steps=40), 
    }

    results = {}
    for attack_name, attack_factory in attack_factories.items():
        print(f'Testing on {attack_name}')
        clean_acc, adv_acc = test_robustness(
            args.trained_model, attack_factory=attack_factory, test_loader=test_loader
        )
        results[attack_name] = (clean_acc, adv_acc)


    # Write to log file
    model_dir, model_file = os.path.split(args.trained_model)
    log_file_name = '.'.join([ model_file.split('.')[0], 'txt' ])

    res_text = '\n'.join([ f'{attack_name}: clean acc: {val[0]:10.8f} adv acc: {val[1]:10.8f}' for attack_name, val in results.items() ])

    with open(os.path.join(model_dir, log_file_name), 'w') as w:
        w.write(res_text)

    print(res_text)


if __name__ == '__main__':
    main()