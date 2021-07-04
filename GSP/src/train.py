import copy
import os.path as osp
from torch.autograd import Variable
from utils import op_copy, cal_acc, lr_schedule
import torch.nn as nn
import torch.optim as optim
import network
from torch.nn import functional as F
import torch
import math

from src.GW_alignment import node_alignment, edge_alignment

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def train(args, dataloader, criterion_classifier_source, criterion_classifier_target, criterion_em_target, criterion):

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=args.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netCs = network.source_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netCt = network.target_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    if args.pretrain:
        modelpath = args.output_dir_src + 'source_B.pt'
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + 'source_Cs.pt'
        netCs.load_state_dict(torch.load(modelpath))
        netCs.eval()
        modelpath = args.output_dir_src + 'source_Ct.pt'
        netCt.load_state_dict(torch.load(modelpath))
        for k, v in netCs.named_parameters():
            v.requires_grad = False

        param_group = []
        for k, v in netB.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False

        for k, v in netCt.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False
    else:
        param_group = []
        for k, v in netB.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

        for k, v in netCs.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

        for k, v in netCt.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_len = max(len(dataloader["source"]), len(dataloader["target"]))
    max_iter = args.max_epoch * max_len
    interval_iter = max_iter // 50
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_source, labels_source, _ = iter_source.next()
        except:
            iter_source = iter(dataloader["source"])
            inputs_source, labels_source, _ = iter_source.next()

        try:
            inputs_target, _, _ = iter_target.next()
        except:
            iter_target = iter(dataloader["target"])
            inputs_target, _, _ = iter_target.next()

        if inputs_source.size(0) == 1 or inputs_target.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netB.eval()
            netCs.eval()
            netCt.eval()
            acc_src = cal_acc(dataloader['tgt_test'], netB, netCs)
            acc_tgt = cal_acc(dataloader['tgt_test'], netB, netCt)
            log_str = 'Task: {} to {}, Iter:{}/{}; src_acc = {:.2f}%, tgt_acc = {:.2f}%'.format(args.name_src, args.name_tgt, iter_num, max_iter, acc_src, acc_tgt)
            print(log_str + '\n')

        
        iter_num += 1
        lr_schedule(optimizer, iter_num=iter_num, max_iter=max_iter)

        labels_source_temp = labels_source + args.class_num
        labels_source_temp = labels_source_temp.cuda()
        labels_source_temp_var = torch.autograd.Variable(labels_source_temp).long()

        inputs_source = inputs_source.cuda()
        inputs_target = inputs_target.cuda()
        labels_source = labels_source.cuda()
        inputs_source_var = torch.autograd.Variable(inputs_source)
        inputs_target_var = torch.autograd.Variable(inputs_target)
        labels_source_var = torch.autograd.Variable(labels_source).long()

        feat_source = netB(inputs_source_var)
        output_src_Cs = netCs(feat_source)
        output_src_Ct = netCt(feat_source)

        feat_target = netB(inputs_target_var)
        temp_feat = feat_target
        temp_tgt_Cs = netCs(temp_feat)
        temp_tgt_Ct = netCt(temp_feat)
        temp_tgt = torch.cat((temp_tgt_Cs, temp_tgt_Ct), 1)
        loss_align = args.alpha * edge_alignment(feat_source, feat_target) + (1.0-args.alpha) * \
                     node_alignment(feat_source, feat_target)[0]



        feat_target = torch.mm(node_alignment(feat_source, feat_target)[1], feat_target)
        output_tgt_Cs = netCs(feat_target)
        output_tgt_Ct = netCt(feat_target)

        output_source = torch.cat((output_src_Cs, output_src_Ct), 1)
        output_target = torch.cat((output_tgt_Cs, output_tgt_Ct), 1)

        loss_task_s_Cs = criterion(output_src_Cs, labels_source_var)
        loss_task_s_Ct = criterion(output_src_Ct, labels_source_var)

        loss_domain_st_Cst_part1 = criterion_classifier_source(output_source)
        loss_category_st_G = 0.5 * criterion(output_source, labels_source_var) + 0.5 * criterion(output_source, labels_source_temp_var)

        loss_domain_st_Cst_part2 = criterion_classifier_target(output_target)
        loss_domain_st_G = 0.5 * criterion_classifier_target(output_target) + 0.5 * criterion_classifier_source(output_target)
        loss_target_em = criterion_em_target(temp_tgt)

        lam = 2 / (1 + math.exp(-1 * 10 * int(iter_num) / max_len)) - 1

        loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2
        loss_G = loss_category_st_G + lam * (loss_domain_st_G + loss_target_em) + args.beta * loss_align


        optimizer.zero_grad()
        loss_classifier.backward(retain_graph=True)
        temp_grad = []
        for param in netB.parameters():
            temp_grad.append(param.grad.data.clone())
        for param in netCs.parameters():
            temp_grad.append(param.grad.data.clone())
        for param in netCt.parameters():
            temp_grad.append(param.grad.data.clone())

        grad_for_classifier = temp_grad

        optimizer.zero_grad()
        loss_G.backward()
        temp_grad = []
        for param in netB.parameters():
            temp_grad.append(param.grad.data.clone())
        for param in netCs.parameters():
            temp_grad.append(param.grad.data.clone())
        for param in netCt.parameters():
            temp_grad.append(param.grad.data.clone())
        grad_for_featureExtractor = temp_grad

        count = 0
        for param in netB.parameters():
            temp_grad = param.grad.data.clone()
            temp_grad.zero_()
            temp_grad = temp_grad + grad_for_featureExtractor[count]
            temp_grad = temp_grad
            param.grad.data = temp_grad
            count = count + 1

        for param in netCs.parameters():
            temp_grad = param.grad.data.clone()
            temp_grad.zero_()
            temp_grad = temp_grad + grad_for_classifier[count]
            temp_grad = temp_grad
            param.grad.data = temp_grad
            count = count + 1

        for param in netCt.parameters():
            temp_grad = param.grad.data.clone()
            temp_grad.zero_()
            temp_grad = temp_grad + grad_for_classifier[count]
            temp_grad = temp_grad
            param.grad.data = temp_grad
            count = count + 1

        optimizer.step()

