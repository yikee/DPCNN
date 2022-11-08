import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable

from utils import *
import shutil
import json
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter
from model import DPCNN
import pickle

def training(train_loader, test_loader, opt):
    net = DPCNN(opt)

    best_acc = 0

    NUM_TRAIN = opt.BATCH_SIZE
    PRE_EPOCH = 0
    NET_PREFIX = opt.NET_SAVE_PATH + net.model_name + '/'
    print('==> Loading Model ...')

    model_name = opt.MODEL_NAME + ".pth"
    model_config = opt.MODEL_NAME + ".cfg"
    opt_save_path = opt.MODEL_NAME + ".opt"
    pickle.dump(opt, open(opt.NET_SAVE_PATH + '/' + opt_save_path, "wb"))
    if not os.path.exists(NET_PREFIX):
        os.mkdir(NET_PREFIX)
    shutil.copyfile("config.py", NET_PREFIX + model_config)
    if not os.path.exist('./source/log/' + net.model_name):
        os.mkdir('/source/log' + net.model_name)
    if os.path.exists(NET_PREFIX + model_name) and opt.RE_TRAIN == False:
        try:
            net, PRE_STEP, best_acc = net.load(NET_PREFIX + model_name)
            print("Load existing model: %s" % (NET_PREFIX + model_name))
        except IOError:
            pass
    
    if opt.USE_CUDA:
        net.cuda()

    criterion = nn.crossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.LEARNING_RATE)
    writer = SummaryWriter(opt.SUMMARY_PATH)
    
    print("Now Tensorboard running. The summary directory is %s" % opt.SUMAMRY_PATH)
    for step, data in enumerate(train_loader):
        PRE_STEP = step
        train_loss = 0
        train_acc = 0
        net.train()
        labels, inputs = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicts = torch.max(outputs, 1)
        num_correct = (predicts == labels).sum()
        train_acc += num_correct.data.item()

        writer.add_scalar("Train/loss", train_loss/NUM_TRAIN, step+PRE_STEP)
        writer.add_scalar("Train/acc", float(train_acc)/NUM_TRAIN, step+PRE_STEP)

        if step % opt.TEST_STEP == 0 and step != 0:
            test_loss, test_acc, topn_acc, news_acc, daily_acc, geng_acc = testing(test_loader, net, opt)
            writer.add_scaler("Test/loss", test_loss, step+PRE_STEP)
            writer.add_sclar("Test/acc", test_acc, step+PRE_STEP)

            if test_acc > best_acc:
                best_acc = test_acc
                net.save((step + PRE_EPOCH), best_acc, model_name)
            print("step is %s; test_loss is %s; test_acc os %s; news_acc is %s; daily_acc is %s; geng_acc is %s;"
                    % (step, test_loss, test_acc, news_acc, daily_acc, geng_acc))
    
    print("==> Training Finished. Current Model is %s. The highest test acc is %.4f" % (net.model_name, best_acc))
    return net
    
def testing(test_loader, net, opt):
    NUM_TEST = len(test_loader.dataset)
    test_loss = 0
    test_acc = 0
    topn_acc = [0] * opt.TOP_NUM
    criterion = nn.CrossEntropyLoss()
    if opt.USE_CUDA:
        net.cuda()

    net.eval()

    news_count = 0
    daily_count = 0
    geng_count = 0

    news_acc = 0
    daily_acc = 0
    geng_acc = 0

    for i, data in enumerate(test_loader):
        labels, inputs = data

        news_count += len(labels[labels == 0])
        daily_count += len(labels[labels == 1])
        geng_count += len(labels[labels == 2])

        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.data.item()
        _, predicts = torch.max(outputs, 1)
        if opt.USE_CUDA:
            labels = labels.cpu().data
            predicts = predicts.cpu().data
        else:
            labels = labels.data
            predicts = predicts.data

        num_correct = (predicts == labels).sum()

        pred = predicts.tolist()
        label = labels.tolist()

        news_num_correct = 0
        daily_num_correct = 0
        geng_num_correct = 0

        for i in range(len(pred)):
            if pred[i] == label[i]:
                if pred[i] == 0: news_num_correct += 1
                if pred[i] == 1: daily_num_correct += 1
                if pred[i] == 2: geng_num_correct += 1
        
        for i in range(opt.TOP_NUM):
            prediction = np.array(outputs.cpu().data.sort(descending=True, dim=1)[1])[:, :(i+1)]
            for j in range(len(labels)):
                if labels[j] in prediction[j]:
                    topn_acc[i] += 1
        
        test_loss += loss.data.item()
        test_acc += num_correct

        news_acc += news_num_correct
        daily_acc += daily_num_correct
        geng_acc += geng_num_correct

        test_loss = float(test_loss) / NUM_TEST
        test_acc = float(test_acc) / NUM_TEST
        topn_acc = [float(x) / NUM_TEST for x in topn_acc]

        news_acc = float(news_acc) / news_count
        daily_acc = float(daily_acc) / daily_count
        geng_acc = float(geng_acc) / geng_count

        return test_loss, test_acc, topn_acc, news_acc, daily_acc, geng_acc