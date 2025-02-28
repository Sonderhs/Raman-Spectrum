import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from utils import *
from config import args
from layer import *


# KFold Cross Validation
def KFold_CV(data_path):
    v = pd.read_csv(data_path).values
    x = v[:, :args.d]
    y = v[:, -1]
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    if args.c == 2:
        df = pd.DataFrame(columns=['AUC', 'Sn', 'Sp', 'Pre', 'Acc', 'F1', 'Mcc'])
    else:
        df = pd.DataFrame(columns=['Rec', 'Pre', 'Acc', 'F1'])
    # tid: training index, vid: validation index
    for i, (tid, vid) in enumerate(kf.split(x)):
        df.loc[i] = train(tid, vid, data_path, i)
    print(df)
    print('mean')
    print(df.mean())
    print('std')
    print(df.std())


def data_loader(tid, vid, data_path):
    v = pd.read_csv(data_path).values
    x = v[:, :args.d]
    y = v[:, -1]
    # train and valid split
    x_train, x_val = x[tid], x[vid]
    y_train, y_val = y[tid], y[vid]
    adj_train = norm_adj(x_train)
    adj_val = norm_adj(x_val)
    x_train = torch.from_numpy(x_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_train = torch.LongTensor(y_train)
    if args.cuda:
        x_train = x_train.cuda()
        x_val = x_val.cuda()
        y_train = y_train.cuda()
        adj_train = adj_train.cuda()
        adj_val = adj_val.cuda()
    return x_train, x_val, y_train, y_val, adj_train, adj_val



def train(tid, vid, data_path, tag=1):
    if args.model == 'GCN':
        model = GNet()
        x_train, x_val, y_train, y_val, adj_train, adj_val = data_loader(tid, vid, data_path)
        print(f"x_train.shape:{x_train.shape}")
        print(f"adj_train.shape:{adj_train.shape}")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss_f = F.cross_entropy

        print('Fold ', tag)
        for e in range(args.epochs):
            model.train()
            pred = model(adj_train, x_train)
            loss = loss_f(pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

            best_acc = 0
            acc = accuracy_score(y_train, pred.argmax(dim=-1))
            if e % 20 == 0 and e != 0:
                print('Epoch %d | Loss: %.4f | Acc: %.4f' % (e, loss.item(), acc))
                # save model
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), './save_model/gcn_best_model.pth')

        model.eval()
        with torch.no_grad():
            pred = model(adj_val, x_val)
            if args.cuda: pred = pred.cpu()
            if args.c == 2:
                predict = pred[:, 1].detach().numpy().flatten()
                res = binary(y_val, predict)
            else:
                res = [
                    recall_score(y_val, pred.argmax(dim=-1), average="weighted"),
                    precision_score(y_val, pred.argmax(dim=-1), average="weighted"),
                    accuracy_score(y_val, pred.argmax(dim=-1)),
                    f1_score(y_val, pred.argmax(dim=-1), average="weighted"),
                ]
        return res


    if args.model == 'CNN' or args.model == 'LSTM':
        if args.model == 'CNN':
            model = CNNNet()
        elif args.model == 'LSTM':
            model = LSTMNet()
        x_train, x_val, y_train, y_val, _, _ = data_loader(tid, vid, data_path)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        x_train = x_train.unsqueeze(1)
        x_val = x_val.unsqueeze(1)
        y_train = torch.LongTensor(y_train)

        if args.cuda:
            model = clf.cuda()
            x_train = x_train.cuda()
            x_val = x_val.cuda()
            y_train = y_train.cuda()

        print('Fold ', tag)
        for e in range(args.epochs):
            model.train()
            z = model(x_train)
            loss = F.cross_entropy(z, y_train)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if e % 20 == 0 and e != 0:
                print('Epoch %d | Loss: %.4f' % (e, loss.item(), acc))
                # save model
                if acc > best_acc:
                    best_acc = acc
                    # if args.model == 'CNN':
                    #     torch.save(model.state_dict(), './save_model/cnn_best_model.pth')
                    # elif args.model == 'LSTM':  
                    #     torch.save(model.state_dict(), './save_model/lstm_best_model.pth')

        model.eval()
        mat = model(x_val)
        if args.cuda: mat = mat.cpu()
        if args.c == 2:
            predict = mat[:, 1].detach().numpy().flatten()
            res = binary(y_val, predict)
        else:
            predict = mat.argmax(dim=-1)
            res = [
                recall_score(y_val, predict, average="weighted"),
                precision_score(y_val, predict, average="weighted"),
                accuracy_score(y_val, predict),
                f1_score(y_val, predict, average="weighted"),
            ]
        return res


def predict(model_path, predict_data_path):
    model = GNet() 
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() 

    pred_data = pd.read_csv(predict_data_path, header=None).values
    print(f"pred_data.shape:{pred_data.shape}")
    prediction_result = []
    for i in range(pred_data.shape[0]):
        x_pred = pred_data[i:i+1, :args.d] 
        adj_pred = norm_adj(x_pred)
        x_pred = torch.from_numpy(x_pred).float()
        if args.cuda:
            x_pred = x_pred.cuda()
            adj_pred = adj_pred.cuda()

        with torch.no_grad():
            pred = model(adj_pred, x_pred)
            prediction = pred.argmax().item()
            prediction_result.append(prediction)
    print(prediction_result)

