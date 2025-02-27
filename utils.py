import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, \
    matthews_corrcoef, roc_auc_score, roc_curve


def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)


def specificity(y_true, y_pred):
    tn, fp = confusion_matrix(y_true, y_pred)[0]
    if tn + fp != 0:
        return tn / (tn + fp)
    else:
        return 1


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


# 评估二分类模型的性能。
# 它接收真实标签 yv 和预测分数 predict，并返回一系列评估指标。
def binary(yv, predict):
    fpr, tpr, th = roc_curve(yv, predict)
    pred = np.ones(len(predict))
    for i in range(len(predict)):
        if predict[i] < th[np.argmax(tpr - fpr)]: pred[i] = 0.0

    confusion = confusion_matrix(yv, pred)

    print(confusion)
    res = [
        roc_auc_score(yv, predict),  # ROC-AUC: ROC曲线下的面积
        sensitivity(yv, pred),  # 灵敏度
        specificity(yv, pred),  # 特异性
        precision_score(yv, pred),  # 精确度
        accuracy_score(yv, pred),  # 准确度
        f1_score(yv, pred),  # F1分数
        matthews_corrcoef(yv, pred)  # 马修斯相关系数
    ]
    return res

# 计算邻接矩阵
def neighborhood(feat, k, spec_ang=False):
    # compute C
    # 计算特征矩阵的内积
    featprod = np.dot(feat.T, feat)
    # 计算特征矩阵的模
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    if spec_ang:
        dmat = 1 - featprod / np.sqrt(smat * smat.T)  # 1 - spectral angle
    else:
        dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


# 计算权重矩阵W= D^-0.5 * W * D^-0.5
def normalized(wmat):
    # 计算度矩阵D
    deg = np.diag(np.sum(wmat, axis=0))
    # D^-0.5
    degpow = np.power(deg, -0.5)
    # 函数将 degpow 中的无穷大值替换为 0,用于处理孤立节点（度为0）
    degpow[np.isinf(degpow)] = 0
    # W= D^-0.5 * W * D^-0.5
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


# 计算归一化邻接矩阵
def norm_adj(feat):
    C = neighborhood(feat.T, k=6, spec_ang=False)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g
