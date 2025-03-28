# -*- coding: utf-8 -*-
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import argparse
import torch
import torch.nn as nn
from model_queue import FusDreamer
import numpy as np
from datasets import get_dataset
import Cimy_PPtools

np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser(description='PyTorch FusDreamer')
parser.add_argument('--source_name', choices=['Trento', 'Houston13', 'Houston18', 'MUUFL', 'GRSS07'], type=str, default='Trento', help='the name of the source dir')
parser.add_argument('--cuda', type=int, default=1, help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--seed', type=int, default=3667, metavar='S', help='random seed ')
group_train = parser.add_argument_group('Training')
group_train.add_argument('--num_eval', type=int, default=5, help='the number of save model')
group_train.add_argument('--num_epoch', type=int, default=100, help='the number of epoch')
group_train.add_argument('--batch_size', type=int, default=64, help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--numComponents', type=int, default=15, help="numComponents (optional, if absent will be set by the model")
group_train.add_argument('--patch_size', type=int, default=33, help="Size of the spatial neighbourhood (optional, if absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-3, help="Learning rate, set by the model if not specified.")
group_train.add_argument('--lambda_1', type=float, default=1e+0, help="Regularization parameter, balancing the alignment loss.")
group_train.add_argument('--alpha', type=float, default=0.3, help="Regularization parameter, controlling the contribution of both coarse-and fine-grained linguistic features.")

args = parser.parse_args()

if __name__ == '__main__':
    if args.source_name == "GRSS07":
        args.numComponents = 5

    save_path = './models/%s/' % args.source_name

    data_root = './datasets/' + args.source_name+'_ps%d' % args.patch_size+'_nc%d' % args.numComponents+'.h5'

    data_loader, sample_index, num_sample, num_classes = Cimy_PPtools.load_fusion_data(data_root, args)

    LABEL_VALUES_src, LABEL_QUEUE = get_dataset(args.source_name)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': args.numComponents, 'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})

    print('Hyperparams Setting:', hyperparams)
    print("Train samples:%d," % len(data_loader[0].dataset), num_sample[0])

    correct = 0

    pretrained_dict = torch.load('./ViT-B-32_torch110.pt', map_location="cpu")
    embed_dim = pretrained_dict["text_projection"].shape[1]
    context_length = pretrained_dict["positional_embedding"].shape[0]
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 3

    model = FusDreamer(embed_dim=embed_dim,
                       dif_inchannel=args.numComponents,
                       inchannel=48,
                       vision_patch_size=hyperparams['patch_size'],
                       num_classes=num_classes,
                       context_length=context_length,
                       vocab_size=vocab_size,
                       transformer_width=transformer_width,
                       transformer_heads=transformer_heads,
                       transformer_layers=transformer_layers,
                       device=DEVICE).to(DEVICE)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Training Parameters: %.6f M.' % (total_trainable_params / (1024 * 1024)))

    criterion = nn.CrossEntropyLoss()
    model, train_time = Cimy_PPtools.train(model, criterion, DEVICE, data_loader[0], args, args.num_epoch,data_loader[1], LABEL_VALUES_src, LABEL_QUEUE)

    test_acc_temp, test_loss_temp, correct, label, test_time = Cimy_PPtools.test(model, criterion, DEVICE, data_loader[2], LABEL_VALUES_src)

    print('=====Classification Accuracy Matrix=====')
    classification, confusion, accuracy_matrix = Cimy_PPtools.reports(label[0], label[1])

    del model
