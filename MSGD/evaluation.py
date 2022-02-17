from model import model
import numpy as np
from sklearn import svm
import torch
from utils.utils import to_cuda, accuracy


# Compute A-distance using numpy and sklearn
# Reference: Analysis of representations in domain adaptation, NIPS-07.
def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def classification_test(loader, net, domain_id):
    res = {}
    res['gt'], res['probs'] = [], []
    net.cuda()
    with torch.no_grad():
        net.set_bn_domain(domain_id)
        for sample in iter(loader):
            img = to_cuda(sample['Img'])
            probs = net(img)['probs']
            res['probs'] += [probs]
            label = to_cuda(sample['Label'])
            res['gt'] += [label]

        probs = torch.cat(res['probs'], dim=0)
        gts = torch.cat(res['gt'], dim=0)
        acc = accuracy(probs, gts) / 100.0
        test_error = 1 - acc
        return acc, test_error


def evaluate(src_feats, tgt_feats, tgt_loader, val_loader, model):

    #computer the a-distance
    a_dis = proxy_a_distance(src_feats, tgt_feats)

    #computer source classifier error
    _, src_test_error = classification_test(val_loader, model, 0)

    test_error = src_test_error + a_dis

    #report the target classification accuracy
    tgt_acc, _ = classification_test(tgt_loader, model, 1)

    print(f"Target Classification Accuracy {tgt_acc:.4f} with test erros {test_error} on validation.")






