import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn

from MSGD_Git.prepare_data import prepare_data_MSGD
from model import model
from prepare_data import *
import sys
import pprint
import warnings
from MSGD_Git.solver.msgd_solver import MSGDSolver as Solver


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--sourcename', dest='sourcename',
                        help='select the source domain', default='Art', type=str)
    parser.add_argument('--targetname', dest='targetname',
                        help='select the source domain', default='RealWorld', type=str)
    parser.add_argument('--dataroot', dest='dataroot',
                        help='path to the images', default='../datasets/office_home/', type=str)
    parser.add_argument('--saveroot', dest='saveroot',
                        help='path to the ckpts', default='../ckpt', type=str)
    parser.add_argument('--num_classes', dest='num_classes', 
                        help='number of categories', default=65, type=int)
    parser.add_argument('--cluster_batch', dest='cluster_batch', 
                        help='batch size for clustering process', default=100, type=int)
    parser.add_argument('--train_batch', dest='train_batch', 
                        help='batch size for training process', default=30, type=int)
    parser.add_argument('--train_class_batch', dest='train_class_batch', 
                        help='batch size for training process in each selected class', default=3, type=int)
    parser.add_argument('--test_batch', dest='test_batch', 
                        help='batch size for testing process', default=30, type=int)
    parser.add_argument('--num_selected_classes', dest='num_selected_classes', 
                        help='number of selected classes in each batch', default=10, type=int)
    parser.add_argument('--test_interval', dest='test_interval', 
                        help='test_interval', default=1.0, type=float)
    parser.add_argument('--save_interval', dest='save_interval', 
                        help='save_interval', default=10.0, type=float)
    parser.add_argument('--lr', dest='lr',
                        help='basic learning rate', default=1e-3, type=float)
    parser.add_argument('--em', dest='em', 
                        help='parameter for em', default=1.0, type=float)
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='office_home', type=str)

    args = parser.parse_args()
    return args

def train(args):
    bn_domain_map = {}

    # method-specific setting 
    dataloaders = prepare_data_MSGD(args)
    num_domains_bn = 2

    # initialize model
    model_state_dict = None
    fx_pretrained = True
    resume_dict = None

    net = model.danet(num_classes=args.num_classes, 
                 state_dict=model_state_dict,
                 feature_extractor='resnet50', 
                 frozen=['layer1'], 
                 fx_pretrained=fx_pretrained, 
                 dropout_ratio=(0.0,),
                 fc_hidden_dims=(), 
                 num_domains_bn=num_domains_bn)

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
       net.cuda()

    # initialize solver
    train_solver = Solver(args, net, dataloaders, bn_domain_map=bn_domain_map, resume=resume_dict)

    # train 
    train_solver.solve()
    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()

    args.saveroot = os.path.join(args.saveroot, args.exp_name)

    for c_times in range(10):
        train(args)
