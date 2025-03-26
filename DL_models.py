# -*- coding: utf-8 -*-

import pandas as pd
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Opitimizer.Adam import Adam
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Top_k_Accuracy import Top_k_Accurary
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Algorithm.Classification.LadderNetwork import LadderNetwork
from LAMDA_SSL.Algorithm.Classification.ImprovedGAN import ImprovedGAN
from LAMDA_SSL.Algorithm.Classification.VAT import VAT
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.TemporalEnsembling import TemporalEnsembling
from LAMDA_SSL.Algorithm.Classification.SSVAE import SSVAE
from LAMDA_SSL.Transform.ToImage import ToImage
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.Vision.CIFAR10 import CIFAR10
from LAMDA_SSL.Network.ResNet50 import ResNet50
from LAMDA_SSL.Network.WideResNet import WideResNet
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
from LAMDA_SSL.Scheduler.CosineAnnealingLR import CosineAnnealingLR
import torch
import torch.nn as nn
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler

device = 'cpu'
device1 = 'cuda:0'
labeled_dataloader = LabeledDataLoader(
    batch_size=32, num_workers=6, drop_last=False)
unlabeled_dataloader = UnlabeledDataLoader(num_workers=6, drop_last=False)
valid_dataloader = UnlabeledDataLoader(
    batch_size=32, num_workers=6, drop_last=False)
test_dataloader = UnlabeledDataLoader(
    batch_size=32, num_workers=6, drop_last=False)
optimizer = Adam(lr=3e-4)


def temporalensembling(X_train, y_train, unlabeled_X):
    labeled_X = X_train.reshape(X_train.shape[0], -1, X_train.shape[1])
    labeled_y = y_train
    unlabeled_X = unlabeled_X.reshape(
        unlabeled_X.shape[0], -1, unlabeled_X.shape[1])
    evaluation = {
        'accuracy': Accuracy(),
        'precision': Precision(average='macro'),
        'Recall': Recall(average='macro'),
        'F1': F1(average='macro'),
        'AUC': AUC(multi_class='ovo'),
        'Confusion_matrix': Confusion_Matrix(normalize='true')
    }
    # network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
    transforms = None
    target_transform = None
    transform = ToTensor(dtype='float', image=False)
    unlabeled_transform = ToTensor(dtype='float', image=False)
    test_transform = ToTensor(dtype='float', image=False)
    valid_transform = ToTensor(dtype='float', image=False)
    train_dataset = None
    labeled_dataset = LabeledDataset(transforms=transforms,
                                     transform=transform, target_transform=target_transform)

    unlabeled_dataset = UnlabeledDataset(transform=unlabeled_transform)

    valid_dataset = UnlabeledDataset(transform=valid_transform)
    network = ResNet50(num_classes=2)
    # network = WideResNet(num_classes=2)
    torch.cuda.set_device(-1)
    augmentation = Noise(noise_level=0.01)
    test_dataset = UnlabeledDataset(transform=test_transform)

    scheduler = CosineAnnealingLR(eta_min=0, T_max=400)
    labeled_sampler = RandomSampler(replacement=True, num_samples=32 * (400))
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()
    optimizer = Adam(lr=3e-4)
    # optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
    model = TemporalEnsembling(lambda_u=0.1, warmup=0.4,
                               mu=1, weight_decay=5e-4, ema_decay=0.999,
                               epoch=10, num_it_epoch=400,
                               eval_it=100, device='cuda:0',
                               labeled_dataset=labeled_dataset,
                               unlabeled_dataset=unlabeled_dataset,
                               valid_dataset=valid_dataset,
                               test_dataset=test_dataset,
                               labeled_sampler=labeled_sampler,
                               unlabeled_sampler=unlabeled_sampler,
                               valid_sampler=valid_sampler,
                               test_sampler=test_sampler,
                               labeled_dataloader=labeled_dataloader,
                               unlabeled_dataloader=unlabeled_dataloader,
                               valid_dataloader=valid_dataloader,
                               test_dataloader=test_dataloader,
                               augmentation=augmentation, network=network,
                               optimizer=optimizer, scheduler=scheduler,
                               evaluation=evaluation, verbose=False)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)
    pred_y = model.predict(unlabeled_X)

    return pred_y,pred_y,model


def flexmatch(X_train, y_train, unlabeled_X):
    labeled_X = X_train.reshape(X_train.shape[0], -1, X_train.shape[1])
    labeled_y = y_train
    unlabeled_X = unlabeled_X.reshape(
        unlabeled_X.shape[0], -1, unlabeled_X.shape[1])

    evaluation = {
        'accuracy': Accuracy(),
        'precision': Precision(average='macro'),
        'Recall': Recall(average='macro'),
        'F1': F1(average='macro'),
        'AUC': AUC(multi_class='ovo'),
        'Confusion_matrix': Confusion_Matrix(normalize='true')
    }
    # network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
    transforms = None
    target_transform = None
    transform = ToTensor(dtype='float', image=False)
    unlabeled_transform = ToTensor(dtype='float', image=False)
    test_transform = ToTensor(dtype='float', image=False)
    valid_transform = ToTensor(dtype='float', image=False)
    train_dataset = None
    labeled_dataset = LabeledDataset(transforms=transforms,
                                     transform=transform, target_transform=target_transform)

    unlabeled_dataset = UnlabeledDataset(transform=unlabeled_transform)

    valid_dataset = UnlabeledDataset(transform=valid_transform)
    network = ResNet50(num_classes=2)
    # network = WideResNet(num_classes=2)
    torch.cuda.set_device(-1)
    augmentation = Noise(noise_level=0.01)
    test_dataset = UnlabeledDataset(transform=test_transform)

    scheduler = CosineAnnealingLR(eta_min=0, T_max=400)
    labeled_sampler = RandomSampler(replacement=True, num_samples=32 * (400))
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()

    # optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
    model = FlexMatch(threshold=0.9, lambda_u=1, T=1.0, mu=7, ema_decay=0.999, weight_decay=5e-4,
                      epoch=10, num_it_epoch=400, eval_it=100, device='cuda:0',
                      labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                      valid_dataset=valid_dataset, test_dataset=test_dataset,
                      labeled_sampler=labeled_sampler, unlabeled_sampler=unlabeled_sampler,
                      valid_sampler=valid_sampler, test_sampler=test_sampler,
                      labeled_dataloader=labeled_dataloader, unlabeled_dataloader=unlabeled_dataloader,
                      valid_dataloader=valid_dataloader, test_dataloader=test_dataloader,
                      augmentation=augmentation, network=network, optimizer=optimizer, scheduler=scheduler,
                      evaluation=evaluation, verbose=False)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    train_performance = model.evaluate(labeled_X, labeled_y, valid=False)
    prob_y = model.predict_proba(unlabeled_X)
    pred_y = model.predict(unlabeled_X)
    return pred_y, prob_y,model


def vat(X_train, y_train, unlabeled_X):
    labeled_X = X_train.reshape(X_train.shape[0], -1, X_train.shape[1])
    labeled_y = y_train
    unlabeled_X = unlabeled_X.reshape(
        unlabeled_X.shape[0], -1, unlabeled_X.shape[1])

    evaluation = {
        'accuracy': Accuracy(),
        'precision': Precision(average='macro'),
        'Recall': Recall(average='macro'),
        'F1': F1(average='macro'),
        'AUC': AUC(multi_class='ovo'),
        'Confusion_matrix': Confusion_Matrix(normalize='true')
    }
    # network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
    transforms = None
    target_transform = None
    transform = ToTensor(dtype='float', image=False)
    unlabeled_transform = ToTensor(dtype='float', image=False)
    test_transform = ToTensor(dtype='float', image=False)
    valid_transform = ToTensor(dtype='float', image=False)
    train_dataset = None
    labeled_dataset = LabeledDataset(transforms=transforms,
                                     transform=transform, target_transform=target_transform)

    unlabeled_dataset = UnlabeledDataset(transform=unlabeled_transform)

    valid_dataset = UnlabeledDataset(transform=valid_transform)
    network = ResNet50(num_classes=2)
    # network = WideResNet(num_classes=2)
    torch.cuda.set_device(-1)
    augmentation = Noise(noise_level=0.01)
    test_dataset = UnlabeledDataset(transform=test_transform)

    scheduler = CosineAnnealingLR(eta_min=0, T_max=400)
    labeled_sampler = RandomSampler(replacement=True, num_samples=32 * (400))
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()

    optimizer = Adam(lr=3e-4)
    # optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
    model = VAT(lambda_u=100, lambda_entmin=0.06, eps=6, xi=1e-6, it_vat=1, warmup=0.4, mu=1,
                weight_decay=5e-4, ema_decay=0.999,
                epoch=10, num_it_epoch=400,
                eval_it=100, device='cuda:0',
                labeled_dataset=labeled_dataset,
                unlabeled_dataset=unlabeled_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                labeled_sampler=labeled_sampler,
                unlabeled_sampler=unlabeled_sampler,
                valid_sampler=valid_sampler,
                test_sampler=test_sampler,
                labeled_dataloader=labeled_dataloader,
                unlabeled_dataloader=unlabeled_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                augmentation=augmentation,
                network=network,
                optimizer=optimizer,
                scheduler=scheduler,
                evaluation=evaluation,
                verbose=False
                )
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)


    return pred_y,prob_y,model


def mixmatch(X_train, y_train, unlabeled_X):
    labeled_X = X_train.reshape(X_train.shape[0], -1, X_train.shape[1])
    labeled_y = y_train
    unlabeled_X = unlabeled_X.reshape(
        unlabeled_X.shape[0], -1, unlabeled_X.shape[1])

    evaluation = {
        'accuracy': Accuracy(),
        'precision': Precision(average='macro'),
        'Recall': Recall(average='macro'),
        'F1': F1(average='macro'),
        'AUC': AUC(multi_class='ovo'),
        'Confusion_matrix': Confusion_Matrix(normalize='true')
    }
    # network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
    transforms = None
    target_transform = None
    transform = ToTensor(dtype='float', image=False)
    unlabeled_transform = ToTensor(dtype='float', image=False)
    test_transform = ToTensor(dtype='float', image=False)
    valid_transform = ToTensor(dtype='float', image=False)
    labeled_dataset = LabeledDataset(transforms=transforms,
                                     transform=transform, target_transform=target_transform)

    unlabeled_dataset = UnlabeledDataset(transform=unlabeled_transform)

    valid_dataset = UnlabeledDataset(transform=valid_transform)
    network = ResNet50(num_classes=2)
    # network = WideResNet(num_classes=2)
    torch.cuda.set_device(-1)
    augmentation = Noise(noise_level=0.01)
    test_dataset = UnlabeledDataset(transform=test_transform)
    # optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
    optimizer = Adam(lr=3e-4)
    scheduler = CosineAnnealingLR(eta_min=0, T_max=400)
    labeled_sampler = RandomSampler(replacement=True, num_samples=32 * (400))
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()
    model = MixMatch(alpha=0.5, lambda_u=100, T=0.5, warmup=1 / 64, mu=1,
                     weight_decay=5e-4, ema_decay=0.999,
                     epoch=10, num_it_epoch=400,
                     eval_it=100, device='cuda:0',
                     labeled_dataset=labeled_dataset,
                     unlabeled_dataset=unlabeled_dataset,
                     valid_dataset=valid_dataset,
                     test_dataset=test_dataset,
                     labeled_sampler=labeled_sampler,
                     unlabeled_sampler=unlabeled_sampler,
                     valid_sampler=valid_sampler,
                     test_sampler=test_sampler,
                     labeled_dataloader=labeled_dataloader,
                     unlabeled_dataloader=unlabeled_dataloader,
                     valid_dataloader=valid_dataloader,
                     test_dataloader=test_dataloader,
                     augmentation=augmentation,
                     network=network,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     evaluation=evaluation, verbose=False
                     )
    # print(random_search.cv_results_)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)


    return pred_y,prob_y,model


def laddernetwork(X_train, y_train, unlabeled_X):
    labeled_X = X_train.reshape(X_train.shape[0], -1, X_train.shape[1])
    labeled_y = y_train
    unlabeled_X = unlabeled_X.reshape(
        unlabeled_X.shape[0], -1, unlabeled_X.shape[1])

    evaluation = {
        'accuracy': Accuracy(),
        'precision': Precision(average='macro'),
        'Recall': Recall(average='macro'),
        'F1': F1(average='macro'),
        'AUC': AUC(multi_class='ovo'),
        'Confusion_matrix': Confusion_Matrix(normalize='true')
    }
    # network=WideResNet(num_clases=10,depth=28,widen_factor=2,drop_rate=0)
    transforms = None
    target_transform = None
    transform = ToTensor(dtype='float', image=False)
    unlabeled_transform = ToTensor(dtype='float', image=False)
    test_transform = ToTensor(dtype='float', image=False)
    valid_transform = ToTensor(dtype='float', image=False)
    train_dataset = None
    labeled_dataset = LabeledDataset(transforms=transforms,
                                     transform=transform, target_transform=target_transform)

    unlabeled_dataset = UnlabeledDataset(transform=unlabeled_transform)

    valid_dataset = UnlabeledDataset(transform=valid_transform)
    network = ResNet50(num_classes=2)
    # network = WideResNet(num_classes=2)
    torch.cuda.set_device(-1)
    augmentation = Noise(noise_level=0.01)
    test_dataset = UnlabeledDataset(transform=test_transform)
    optimizer = Adam(lr=3e-4)
    scheduler = CosineAnnealingLR(eta_min=0, T_max=400)
    labeled_sampler = RandomSampler(replacement=True, num_samples=400)
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()

    model = LadderNetwork(noise_std=0.2,
                          lambda_u=[0.1, 0.1, 0.1, 0.1, 0.1, 10., 1000.],
                          dim_encoder=[1000, 500, 250, 250, 250],
                          encoder_activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
                          mu=1, weight_decay=5e-4,
                          epoch=100, num_it_epoch=400, eval_epoch=10,
                          optimizer=optimizer, scheduler=scheduler, evaluation=evaluation, device='cuda:0',
                          labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                          valid_dataset=valid_dataset,
                          test_dataset=test_dataset,
                          labeled_sampler=labeled_sampler, unlabeled_sampler=unlabeled_sampler,
                          valid_sampler=valid_sampler,
                          test_sampler=test_sampler,
                          labeled_dataloader=labeled_dataloader, unlabeled_dataloader=unlabeled_dataloader,
                          valid_dataloader=valid_dataloader, test_dataloader=test_dataloader, verbose=False)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)

    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model


def uda(X_train, y_train, unlabeled_X):
    labeled_X = X_train.reshape(X_train.shape[0], -1, X_train.shape[1])
    labeled_y = y_train
    unlabeled_X = unlabeled_X.reshape(
        unlabeled_X.shape[0], -1, unlabeled_X.shape[1])

    evaluation = {
        'accuracy': Accuracy(),
        'precision': Precision(average='macro'),
        'Recall': Recall(average='macro'),
        'F1': F1(average='macro'),
        'AUC': AUC(multi_class='ovo'),
        'Confusion_matrix': Confusion_Matrix(normalize='true')
    }
    # network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
    transforms = None
    target_transform = None
    transform = ToTensor(dtype='float', image=False)
    unlabeled_transform = ToTensor(dtype='float', image=False)
    test_transform = ToTensor(dtype='float', image=False)
    valid_transform = ToTensor(dtype='float', image=False)
    train_dataset = None
    labeled_dataset = LabeledDataset(transforms=transforms,
                                     transform=transform, target_transform=target_transform)

    unlabeled_dataset = UnlabeledDataset(transform=unlabeled_transform)

    valid_dataset = UnlabeledDataset(transform=valid_transform)
    network = ResNet50(num_classes=2)
    # network = WideResNet(num_classes=2)
    torch.cuda.set_device(-1)
    augmentation = Noise(noise_level=0.01)
    test_dataset = UnlabeledDataset(transform=test_transform)

    scheduler = CosineAnnealingLR(eta_min=0, T_max=400)
    labeled_sampler = RandomSampler(replacement=True, num_samples=32 * (400))
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()
    optimizer = Adam(lr=3e-4)
    # optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
    model = UDA(threshold=0.95, lambda_u=100, T=0.4, mu=7,
                weight_decay=5e-4, ema_decay=0.999,
                epoch=10, num_it_epoch=400,
                eval_it=100, device='cuda:0',
                labeled_dataset=labeled_dataset,
                unlabeled_dataset=unlabeled_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                labeled_sampler=labeled_sampler,
                unlabeled_sampler=unlabeled_sampler,
                valid_sampler=valid_sampler,
                test_sampler=test_sampler,
                labeled_dataloader=labeled_dataloader,
                unlabeled_dataloader=unlabeled_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                augmentation=augmentation,
                network=network,
                optimizer=optimizer,
                scheduler=scheduler,
                evaluation=evaluation, verbose=False

                )
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model