import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, \
    matthews_corrcoef, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize

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

def plot_confusion_matrix(yv, pred, tag):
    confusion = confusion_matrix(yv, pred)
    print(f"Confusion Matrix in Fold {tag}: \n{confusion}")
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for Fold {tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_roc_curve(y_val, pred):
    # 预测测试集的概率  
    y_scores = pred[:, 1].detach().numpy().flatten()
    print(f"y_scores: {y_scores}")
    # 计算ROC曲线的点  
    print(f"y_val: {y_val}")
    print(f"y_scores: {y_scores}")
    fpr, tpr, thresholds = roc_curve(y_val, y_scores)
    # 计算AUC值  
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线  
    plt.figure()
    plt.plot(fpr, tpr, color='#800000', lw=2, label='ROC curve (area = %0.2f)' % roc_auc) # 将曲线颜色改为酒红色
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--') # 虚线颜色保持不变
    # 设置坐标轴标签和图例的字体和大小
    plt.xlabel('False Positive Rate', fontsize=15, fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=15, fontname='Arial')
    # 设置坐标轴的范围
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # 设置坐标轴刻度的字体和大小
    plt.xticks(fontsize=15, fontname='Arial')
    plt.yticks(fontsize=15, fontname='Arial')
    # 显示图表
    plt.show()
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    # 保存到CSV文件
    roc_data.to_csv('roc_curve_XGBoost_magainin2.csv', index=False)

def plot_ovr_roc_curve(y_val, pred):
    n_classes = pred.shape[1]  # 获取类别数量
    y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    plt.figure()
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 3-Class(One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()

# 评估二分类模型的性能。
# 它接收真实标签 yv 和预测分数 predict，并返回一系列评估指标。
def binary(yv, predict, tag):
    fpr, tpr, th = roc_curve(yv, predict)
    pred = np.ones(len(predict))
    for i in range(len(predict)):
        if predict[i] < th[np.argmax(tpr - fpr)]: pred[i] = 0.0
    
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
