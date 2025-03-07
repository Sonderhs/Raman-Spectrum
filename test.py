from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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

        # 计算并打印混淆矩阵
        y_pred = pred.argmax(dim=-1).numpy()
        cm = confusion_matrix(y_val, y_pred)
        print(f"Confusion Matrix for Fold {tag}:")
        print(cm)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for Fold {tag}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()