from prepare_data import generate_dataloader
import argparse
import os, sys
import os.path as osp
import torch
from scipy.io import loadmat
from utils import *
from train import train
import pandas as pd
import numpy as np
import copy
import torch.nn as nn
from SourceClassifier import DClassifierForSource
from TargetClassifier import DClassifierForTarget
from EntropyLoss import EMLossForTarget

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='symnet')
    # data path
    parser.add_argument('--source_root', type=str, default='../dataset/')
    parser.add_argument('--target_root', type=str, default='../dataset/')
    parser.add_argument('--output_dir_src', type=str, default='../ckpts/result/')

    # network parameters
    parser.add_argument('--in_features', type=int, default=2048)
    parser.add_argument('--class_num', type=int, default=65)

    # training parameters
    parser.add_argument('--dset', type=str, default='imageCLEF')
    parser.add_argument('--train_batch', type=int, default=128)
    parser.add_argument('--test_batch', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])


    args = parser.parse_args()

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log_' + args.dset + '.txt'), 'w')

    if args.dset == 'office':
        args.class_num = 31
        args.name_src = 'amazon'
        args.name_tgt = 'dslr'
        args.source_root = '../dataset/' + args.dset + '/' + args.name_src +'_' + args.name_src + '.csv'
        args.target_root = '../dataset/' + args.dset + '/' + args.name_src +'_' + args.name_tgt + '.csv'

        src_data = pd.read_csv(args.source_root, header=None, index_col=None).values
        tgt_data = pd.read_csv(args.target_root, header=None, index_col=None).values

        src_imgs, src_labels = np.float32(src_data[:, :-1]), \
                               np.float32(src_data[:, -1])

        tgt_imgs, tgt_labels = np.float32(tgt_data[:, :-1]), \
                               np.float32(tgt_data[:, -1])

    elif args.dset == 'office-home':
        args.class_num = 65
        args.name_src = 'Art'
        args.name_tgt = 'RealWorld'
        args.source_root = '../dataset/' + args.dset + '/' + args.name_src +'_' + args.name_src + '.csv'
        args.target_root = '../dataset/' + args.dset + '/' + args.name_src +'_' + args.name_tgt + '.csv'

        src_data = pd.read_csv(args.source_root, header=None, index_col=None).values
        tgt_data = pd.read_csv(args.target_root, header=None, index_col=None).values

        src_imgs, src_labels = np.float32(src_data[:, :-1]), \
                               np.float32(src_data[:, -1])

        tgt_imgs, tgt_labels = np.float32(tgt_data[:, :-1]), \
                               np.float32(tgt_data[:, -1])

    elif args.dset=='imageCLEF':
        args.class_num = 12
        args.name_src = 'p'
        args.name_tgt = 'i'
        args.source_root = args.source_root + args.dset + '/' + args.name_src +'_' + args.name_src + '.csv'
        args.target_root = args.target_root + args.dset + '/' + args.name_src +'_' + args.name_tgt + '.csv'

        src_data = pd.read_csv(args.source_root, header=None, index_col=None).values
        tgt_data = pd.read_csv(args.target_root, header=None, index_col=None).values

        src_imgs, src_labels = np.float32(src_data[:, :-1]), \
                              np.float32(src_data[:, -1])

        tgt_imgs, tgt_labels = np.float32(tgt_data[:, :-1]), \
                               np.float32(tgt_data[:, -1])

    dataloader = {}
    dataloader['source'] = generate_dataloader(src_imgs, src_labels, args.train_batch, True, True)
    dataloader['target'] = generate_dataloader(tgt_imgs, tgt_labels, args.train_batch, True, True)
    dataloader['tgt_test'] = generate_dataloader(tgt_imgs, tgt_labels, args.test_batch, False, False)

    criterion_classifier_target = DClassifierForTarget(nClass=args.class_num).cuda()
    criterion_classifier_source = DClassifierForSource(nClass=args.class_num).cuda()
    criterion_em_target = EMLossForTarget(nClass=args.class_num).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    train(args, dataloader, criterion_classifier_source, criterion_classifier_target, criterion_em_target, criterion)



