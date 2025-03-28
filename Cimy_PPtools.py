# -*- coding: utf-8 -*-
# Cimy's Python Pytorch Toolbox
"""
    In this tools, the image show function show the multimodal fusion result.
    If you want to show the single modal classification result need to modify it!

"""
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import random
import copy
import datetime
from prettytable import PrettyTable
from tqdm import tqdm
import clip
import scipy.io as scio
import math
import torch.optim as optim
import time
import h5py
import os
import torch.utils.data as data
from torch.utils.data import TensorDataset


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device

def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def createPatches_TF(X, y, windowSize, removeZeroLabels=False):
    """
        Create the image patches
        Arguments:
             X:                The original input data
             y:                The corresponding label
             windowSize:       Patch window size
             removeZeroLabels: Whether to return the patch results of entire image, default=False.
        Return:
             patchesData:      Patch data
             patchesLabels:    Patch data with corresponding label
    """
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), 'symmetric')
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype='float16')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float16')
    patchIndex = 0
    for c in range(margin, zeroPaddedX.shape[1] - margin):
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1, :]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def createPatches_Torch(X, y, windowSize, removeZeroLabels=False):
    """
        Create the image patches
        Arguments:
             X:                The original input data
             y:                The corresponding label
             windowSize:       Patch window size
             removeZeroLabels: Whether to return the patch results of entire image, default=False.
        Return:
             patchesData:      Patch data
             patchesLabels:    Patch data with corresponding label
    """
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.transpose(np.pad(X, ((margin, margin), (margin, margin), (0, 0)), 'symmetric'), (2, 0, 1))
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], X.shape[2], windowSize, windowSize), dtype='float16')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float16')
    patchIndex = 0
    for c in range(margin, zeroPaddedX.shape[2] - margin):
        for r in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[:, r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData.astype(np.float16), patchesLabels.astype(np.int8)


def random_sample(train_sample, validate_sample, Labels):
    """
        Randomly generate training, validate, and test sets
        Arguments:
             train_sample:                         The vector contains the number of training samples per class
             validate_sample:                      The vector contains the number of validate samples per class
             Labels:                               The ground truth data
        Return:
             TrainIndex, ValidateIndex, TestIndex: The vectorized coordinate values
    """
    num_classes = int(np.max(Labels))
    TrainIndex = []
    TestIndex = []
    ValidateIndex = []
    for i in range(num_classes):
        train_sample_temp = train_sample[i]
        validate_sample_temp = validate_sample[i]
        index = np.where(Labels == (i + 1))[0]
        Train_Validate_Index = random.sample(range(0, int(index.size)), train_sample_temp + validate_sample_temp)
        TrainIndex = np.hstack((TrainIndex, index[Train_Validate_Index[0:train_sample_temp]])).astype(np.int32)
        ValidateIndex = np.hstack((ValidateIndex, index[Train_Validate_Index[train_sample_temp:100000]])).astype(np.int32)
        Test_Index = [index[i] for i in range(0, len(index), 1) if i not in Train_Validate_Index]
        TestIndex = np.hstack((TestIndex, Test_Index)).astype(np.int32)

    return TrainIndex, ValidateIndex, TestIndex


def applyPCA(X, numComponents=75):
    """
        Apply PCA preprocessing for original data
        Arguments:
             X:             The original input data
             numComponents: the hyperparameter of reduced dimension
        Return:
             newX:          Dimensionality reduced data
             pca:
    """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def reports(y_pred, Labels):
    """
        Obtain the final classification accuracy
        Arguments:
             y_pred:           The predict result of models
             Labels:           The ground truth data
        Return:
             Confusion Matrix: Tensor: class x class
             Accuracy matrix:  Vector: (class + 3) x 1
    """
    classification = classification_report(Labels, y_pred)
    confusion = confusion_matrix(Labels, y_pred)
    oa = np.trace(confusion) / sum(sum(confusion))
    ca = np.diag(confusion) / confusion.sum(axis=1)
    Pe = (confusion.sum(axis=0) @ confusion.sum(axis=1)) / np.square(sum(sum(confusion)))
    K = (oa - Pe) / (1 - Pe)
    aa = sum(ca) / len(ca)
    List = []
    List.append(np.array(oa)), List.append(np.array(K)), List.append(np.array(aa))
    List = np.array(List)
    accuracy_matrix = np.concatenate((ca, List), axis=0)
    # ==== Print table accuracy use PrettyTable====
    x = PrettyTable()
    x.add_column('index', [list(range(1, len(ca) + 1, 1)) + ['OA', 'AA', 'KA']][0])
    x.add_column('Accuracy', accuracy_matrix)
    print(x)
    return classification, confusion, accuracy_matrix


def val(model, val_loader, label_name, criterion, device):
    """
        The validate function
        Arguments:
             model:      The trained models
             val_loader: The validate data set
        Return:
             acc:        The accuracy on the validate set
             avg_loss:   The accuracy on the validate set
    """
    global acc, acc_best
    model.eval()
    # total_correct = 0
    eye = torch.eye(int(max(val_loader.dataset.tensors[2]) + 1)).to(device)
    avg_loss = 0.0

    loss = 0
    correct = 0
    loss_coarse = 0
    pred_list, label_list = [], []

    start_time = datetime.datetime.now()
    with tqdm(
            iterable=val_loader,
    ) as t:
        with torch.no_grad():
            for i, (data_hsi, data_lidar, labels) in enumerate(val_loader):
                data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), labels.to(device)

                loss_coarse_, label_src_pred = model(data_hsi, data_lidar,device=device)
                pred = label_src_pred.data.max(1)[1]
                pred_list.append(pred.cpu().numpy())
                label_list.append(target.cpu().numpy())

                target = target.to(torch.int64).to(device)
                target_hot = eye[target]
                loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), target.long()).item()
                loss_coarse += loss_coarse_.item()
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                acc = float(correct) / len(val_loader.dataset.tensors[0])
                cur_time = datetime.datetime.now()
                t.set_description_str(f"\33[39m[  Validation Set  ]")
                t.set_postfix_str(f"Val_Loss = {avg_loss/len(val_loader.dataset.tensors[0]):.6f}, Val_Accuracy = {acc:.6f}, Time: {cur_time - start_time}\33[0m")
                t.update()
        avg_loss /= len(val_loader.dataset.tensors[0])
        acc = float(correct) / len(val_loader.dataset.tensors[0])

    return acc, avg_loss


def train(model, criterion, device, train_loader, args, EPOCHS, val_loader, label_name, label_queue, itera=1):
    """
        The train function
        Arguments:
             model:                                               The constructed models
             criterion:                                           The loss function
             device:                                              Use GPU or CPU for training
             train_loader:                                        The training data set
             optimizer:                                           The optimization function
             EPOCHS:                                              The total training epoch
             vis:                                                 whether to visual the training precessing
             val_loader:                                          The validate data set
             itera:                                               The repeated times of experiments
        Return:
             Model:                                               The trained models
             (end_time_train - start_time_train).total_seconds(): The training time
    """
    global best_model

    acc_temp = 0
    epoch_temp = 1
    loss_temp = 10000
    weight_lambda = []
    weight_alpha = []
    eye = torch.eye(int(max(train_loader.dataset.tensors[2]) + 1)).to(device)
    start_time_train = datetime.datetime.now()
    for epoch in range(1, EPOCHS + 1):
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / EPOCHS), 0.75)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        CNN_correct= 0

        start_time = datetime.datetime.now()
        model.train()
        number = 0
        with tqdm(
                iterable=train_loader,
        ) as t:
            for batch_idx, (data_hsi, data_lidar, target) in enumerate(train_loader):
                t.set_description_str(f"\33[34m[Epoch {epoch:03d}/ {(EPOCHS):03d}/ {(itera):02d}]")

                data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)

                optimizer.zero_grad()
                text = torch.cat(
                    [clip.tokenize(f'A hyperspectral and lidar multimodal data of {label_name[int(k)]}').to(k.device) for k in target])
                text_queue_0 = [label_queue[label_name[int(k)]][0] for k in target]
                text_queue_1 = [label_queue[label_name[int(k)]][1] for k in target]
                text_queue_2 = [label_queue[label_name[int(k)]][2] for k in target]
                text_queue_0 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_0])
                text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
                text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])
                loss_coarse, loss_fine, loss_dif, label_src_pred = model(data_hsi, data_lidar,
                                                                         text=text, label=target,
                                                                         text_queue_0=text_queue_0,
                                                                         text_queue_1=text_queue_1,
                                                                         text_queue_2=text_queue_2,
                                                                         device=device)

                target = target.to(torch.int64).to(device)
                target_hot = eye[target]
                loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), target.long())

                loss = 0.6 * loss_cls + 0.2 * ((1 - 0.2) * loss_coarse + 0.2 * loss_fine) + 0.2 * loss_dif


                loss.backward()
                optimizer.step()

                pred = label_src_pred.data.max(1)[1]
                CNN_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                cur_time = datetime.datetime.now()
                t.set_postfix_str(
                    f"Tra_Loss = {loss.item():.6f}, Tra_Accuracy = {CNN_correct / len(train_loader.dataset):.6f}, Time: {cur_time - start_time}\33[0m")
                t.update()
                if np.isnan(loss.item()):
                    end_time_train = datetime.datetime.now()
                    return model, (end_time_train - start_time_train).total_seconds()
        val_acc, avg_loss = val(model, val_loader, label_name, criterion, device)
        if acc_temp <= val_acc:
            print('Best_Val_Value changed: from %f to %f;' % (acc_temp, val_acc), end="\t")
            epoch_temp = epoch
            acc_temp = val_acc
            best_model = copy.deepcopy(model)
            print('Best Classification Accuracy %f, Best Classification loss %f, Best Epoch %d' % (
            acc_temp, avg_loss, epoch_temp), end="\n")
        else:
            print('Best Classification Accuracy %f, Best Classification loss %f, Best Epoch %d' % (
            acc_temp, avg_loss, epoch_temp), end="\n")

        time.sleep(0.1)
    model = best_model
    end_time_train = datetime.datetime.now()
    print('||======= Train Time for % s' % (end_time_train - start_time_train), '======||')
    return model, (end_time_train - start_time_train).total_seconds()


def test(model, criterion, device, test_loader, label_name):
    """
        The test function
        Arguments:
             model:                                             The constructed models
             device:                                            Use GPU or CPU for training
             test_loader:                                       The test data set
        Return:
             test_acc_temp:                                     The test accuracy
             test_loss_temp:                                    The test loss
             y_pred:                                            The predicted results
             target_1:                                          The ground truth
             (end_time_test - start_time_test).total_seconds(): The test time
    """
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    target_1 = []
    start_time_test = datetime.datetime.now()
    eye = torch.eye(int(max(test_loader.dataset.tensors[2]) + 1)).to(device)
    avg_loss = 0.0

    loss = 0
    correct = 0
    loss_coarse = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for (data_hsi, data_lidar, target) in test_loader:
            data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)

            loss_coarse_, label_src_pred = model(data_hsi, data_lidar, device=device)

            target = target.to(torch.int64).to(device)
            target_hot = eye[target]

            test_loss = F.nll_loss(F.log_softmax(label_src_pred, dim=1), target.long())

            loss_coarse += loss_coarse_.item()

            y_pred_temp = label_src_pred.data.max(1)[1]
            correct += y_pred_temp.eq(target.data.view_as(y_pred_temp)).cpu().sum()
            y_pred_temp_1 = y_pred_temp.data.cpu().numpy()
            target_temp_1 = target.data.cpu().numpy()
            y_pred.extend(y_pred_temp_1)
            target_1.extend(target_temp_1)

        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(1, y_pred.size)
        y_pred = np.array(y_pred).astype(np.float32)
        y_pred = y_pred[0]
    acc = float(correct) / len(test_loader.dataset.tensors[0])
    cur_time = datetime.datetime.now()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc_temp = format(100. * correct / len(test_loader.dataset))
    test_loss_temp = format(test_loss)
    end_time_test = datetime.datetime.now()
    print('||======= Test Time for % s' % (end_time_test - start_time_test), '======||')
    return test_acc_temp, test_loss_temp, correct, [y_pred, target_1], (end_time_test - start_time_test).total_seconds()



def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_fusion_data(data_root, args):
    start_time_test = datetime.datetime.now()

    if os.path.exists(data_root):
        file = h5py.File(data_root, 'r')
        patchesData_hsi = file['patchesData_hsi'][:]
        patchesData_lidar = file['patchesData_lidar'][:]
        # patchesData_RGB = file['patchesData_RGB'][:]
        patchesLabels = file['patchesLabels'][:]

    else:
        if args.source_name == "GRSS07":
            source_data = r'E:\Lab\New\d_demo\2.Feature\a_fusiondatasets\\' + '%s' % args.source_name + '_SAR_MS.mat'
            labels = scio.loadmat(source_data)['ground']
            data_hsi_o = np.array(scio.loadmat(source_data)['HSI_data'])
            if len(np.array(scio.loadmat(source_data)['SAR_data']).shape) == 2:
                data_lidar = np.array(scio.loadmat(source_data)['SAR_data'])
            else:
                data_lidar = np.array(scio.loadmat(source_data)['SAR_data'])[:, :, 0]
        else:
            # source_data = '.\\fusiondatasets\\'  + '%s' % args.source_name + '_hsi_lidar_rgb.mat'
            source_data = r'E:\Lab\New\d_demo\2.Feature\a_fusiondatasets\\'  + '%s' % args.source_name + '_hsi_lidar_rgb.mat'
            labels = scio.loadmat(source_data)['ground']
            data_hsi_o = np.array(scio.loadmat(source_data)['HSI_data'])
            if len(np.array(scio.loadmat(source_data)['LiDAR_data']).shape)==2:
                data_lidar = np.array(scio.loadmat(source_data)['LiDAR_data'])
            else:
                data_lidar = np.array(scio.loadmat(source_data)['LiDAR_data'])[:, :, 0]

        data_hsi, _ = applyPCA(data_hsi_o, numComponents=args.numComponents)
        data_lidar = np.reshape(data_lidar, (data_lidar.shape[0], data_lidar.shape[1], 1))
        data_lidar, _ = applyPCA(data_lidar, numComponents=1)

        patchesData_hsi, _ = createPatches_Torch(data_hsi, labels, windowSize=args.patch_size)
        patchesData_lidar, patchesLabels = createPatches_Torch(data_lidar, labels, windowSize=args.patch_size)

        file = h5py.File(data_root, 'w')
        file.create_dataset('patchesData_hsi', data=patchesData_hsi)
        file.create_dataset('patchesData_lidar', data=patchesData_lidar)
        file.create_dataset('patchesLabels', data=patchesLabels)

    if args.source_name == "Trento":
        # train_sample = (129, 125, 105, 154, 184, 122)
        train_sample = (13, 13, 11, 15, 18, 12)
        validate_sample = (80, 80, 80, 80, 80, 80)

    elif args.source_name == "Houston13":
        # train_sample = (20, 19, 19, 19, 19, 18, 20, 19, 19, 19, 18, 19, 18, 18, 19)  # Baseline
        train_sample = (198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187)        # Baseline
        # train_sample = (188, 188, 105, 187, 186, 49, 190, 187, 188, 184, 185, 185, 70, 64, 99)  # 15%
        # train_sample = ( 313,  314, 174, 311, 311,  81,  317, 311, 313,  307, 309, 308, 117, 107, 165)    # 25%
        # train_sample = ( 438,  439, 244, 435, 435, 114,  444, 435, 438,  429, 432, 432, 164, 150, 231)    # 35%
        # train_sample = ( 626,  627, 349, 622, 621, 163,  634, 622, 626,  614, 618, 617, 235, 214, 330)    # 50%
        # train_sample = (1001, 1003, 558, 995, 994, 260, 1014, 995, 1002, 982, 988, 986, 375, 342, 528)    # 80%
        validate_sample = (80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80)
        # validate_sample = (60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60)
    elif args.source_name == "Houston18":
        # train_sample = (500, 500, 68, 500, 500, 451, 26, 500, 500, 500, 500, 151, 500, 500, 500, 500, 14, 500, 500, 500)
        train_sample = (50, 50, 7, 50, 50, 45, 3, 50, 50, 50, 50, 15, 50, 50, 50, 50, 2, 50, 50, 50)
        validate_sample = (80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80)
    elif args.source_name == "MUUFL":
        train_sample = (15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15)
        # train_sample = (150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150)
        validate_sample = (80, 80, 80, 80, 80, 80, 80, 80, 80, 10, 80)
    elif args.source_name == "GRSS07":
        train_sample = (674, 1481, 752, 45, 30)
        # train_sample = (67, 148, 75, 5, 3)
        validate_sample = (80, 80, 80, 80, 80)

    num_classes = np.max(np.max(patchesLabels))

    trainIndex, valIndex, testIndex = random_sample(train_sample, validate_sample, patchesLabels)

    true_label = patchesLabels[testIndex]

    x_train_hsi, x_test_hsi, x_val_hsi = np.array(patchesData_hsi[trainIndex, :, :, :]).astype(np.float32), \
                                         np.array(patchesData_hsi[testIndex, :, :, :]).astype(np.float32), \
                                         np.array(patchesData_hsi[valIndex, :, :, :]).astype(np.float32)
    x_train_lidar, x_test_lidar, x_val_lidar = np.array(patchesData_lidar[trainIndex, :, :, :]).astype(np.float32), \
                                               np.array(patchesData_lidar[testIndex, :, :, :]).astype(np.float32), \
                                               np.array(patchesData_lidar[valIndex, :, :, :]).astype(np.float32)

    y_train, y_val, y_test = np.array(patchesLabels[trainIndex] - 1).astype(np.float32), \
                             np.array(patchesLabels[valIndex] - 1).astype(np.float32), \
                             np.array(patchesLabels[testIndex] - 1).astype(np.float32)

    x_train_hsi_torch, x_val_hsi_torch, x_test_hsi_torch = torch.from_numpy(x_train_hsi), \
                                                           torch.from_numpy(x_val_hsi), \
                                                           torch.from_numpy(x_test_hsi)
    x_train_lidar_torch, x_val_lidar_torch, x_test_lidar_torch = torch.from_numpy(x_train_lidar), \
                                                                 torch.from_numpy(x_val_lidar), \
                                                                 torch.from_numpy(x_test_lidar)

    y_train_torch, y_val_torch, y_test_torch = torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(
        y_test)

    train_loader = data.DataLoader(dataset=TensorDataset(x_train_hsi_torch, x_train_lidar_torch, y_train_torch),
                                   batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=TensorDataset(x_val_hsi_torch, x_val_lidar_torch, y_val_torch),
                                 batch_size=args.batch_size, shuffle=True)

    test_loader = data.DataLoader(dataset=TensorDataset(x_test_hsi_torch, x_test_lidar_torch, y_test_torch),
                                  batch_size=args.batch_size)

    end_time_test = datetime.datetime.now()

    if os.path.exists(data_root):
        print('||== Load Preprocessed Data \'%s\' using %.4f s ==||' % (data_root, (end_time_test - start_time_test).total_seconds()))
    else:
        print('||== New preprocessed data has been saved into: %s' % data_root)
        print('||== Load Preprocessed Data \'%s\' using %.4f s ==||' % (data_root, (end_time_test - start_time_test).total_seconds()))

    return [train_loader, val_loader, test_loader], [trainIndex, valIndex, testIndex], [train_sample, validate_sample], num_classes

