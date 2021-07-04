import os
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader

def prepare_data_MSGD(args):
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = args.sourcename
    target = args.targetname
    dataroot_S = os.path.join(args.dataroot, source)
    dataroot_T = os.path.join(args.dataroot, target)

    with open(os.path.join(args.dataroot, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == args.num_classes)

    # for clustering
    batch_size = args.cluster_batch
    dataset_type = 'SingleDataset'
    dataloaders['clustering_' + source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=1,
                classnames=classes)

    batch_size = args.cluster_batch
    dataset_type = 'SingleDatasetWithoutLabel'
    dataloaders['clustering_' + target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=1,
                classnames=classes)

    # class-agnostic source dataloader for supervised training
    batch_size = args.train_batch
    dataset_type = 'SingleDataset'
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=1, 
                classnames=classes)

    dataloaders[target] = CustomDatasetDataLoader(
        dataset_root=dataroot_T, dataset_type=dataset_type,
        batch_size=batch_size, transform=train_transform,
        train=True, num_workers=1,
        classnames=classes)

    # initialize the categorical dataloader
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = args.train_class_batch
    target_batch_size = args.train_class_batch
    dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type, 
                source_batch_size=source_batch_size, 
                target_batch_size=target_batch_size, 
                source_dataset_root=dataroot_S, 
                transform=train_transform, 
                classnames=classes, 
                num_workers=1,
                drop_last=True, sampler='RandomSampler')

    batch_size = args.test_batch
    dataset_type = 'SingleDataset'
    test_domain = target
    dataroot_test = os.path.join(args.dataroot, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=1,
                    classnames=classes)

    return dataloaders