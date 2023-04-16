import torch
import data_process
import model
from transformers import AdamW
import pandas as pd
import numpy as np
import json
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.metrics import classification_report
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

#每次初始化
combinemodel = model.CombineModel()
myModel = combinemodel
# myModel.load_state_dict(torch.load('mean_param.pth'))
myModel.to(device)

for param in combinemodel.parameters():
    param.requires_grad_(True)
for param in model.pretrained.parameters():
    param.requires_grad_(False)


#optimizer = AdamW(combineModel.parameters(), lr=5e-4)
optimizer = AdamW(myModel.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
commentf = './programming.txt'
result_f = './test.txt'

acc_list = []
loss_list = []
precision_list = []
recall_list = []
f1_list = []



def train(datadir):
    print("train:")
    combinemodel = model.CombineModel()
    myModel = combinemodel
    # myModel.load_state_dict(torch.load('mean_param.pth'))
    myModel.to(device)
    for param in combinemodel.parameters():
        param.requires_grad_(True)
    for param in model.pretrained.parameters():
        param.requires_grad_(False)

    # optimizer = AdamW(combineModel.parameters(), lr=5e-4)
    optimizer = AdamW(myModel.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()


    # 数据加载
    data_f = "train.csv"
    loader = data_process.dataprocess(data_f, datadir)

    myModel.train()
    for epoch in range(3):
        acc = []
        loss_list = []
        label_true = []
        label_pred = []
        acc_all = 0
        loss_all = 0
        print("epoch数：" + str(epoch + 1))
        for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(loader):
            # print(input_ids)
            # print(labels)
            if (torch.cuda.is_available()):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device), labels.to(device)
            out = myModel(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
            # print(out)
            # print(feature.shape)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 5 == 0:
                # print(out)
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                # print(out)
                # print(labels)
                label_pred = label_pred + out.tolist()
                label_true = label_true + labels.tolist()
                acc.append(accuracy)
                loss_list.append(float(loss))
                print(i, loss.item(), accuracy)

            if i == 670:  # 2000条
                for num in range(len(acc)):
                    acc_all = acc_all + acc[num]
                    loss_all = loss_all + loss_list[num]
                acc_all = acc_all / len(acc)
                loss_all = loss_all / len(loss_list)
                print(loss_all, acc_all)
                measure_result = classification_report(label_true, label_pred)
                print('measure_result = \n', measure_result)
                print("accuracy：%.2f" % accuracy_score(label_true, label_pred))
                print("precision：%.2f" % precision_score(label_true, label_pred))
                print("recall：%.2f" % recall_score(label_true, label_pred))
                print("f1-score：%.2f" % f1_score(label_true, label_pred))
                with open(commentf, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(epoch + 1, ensure_ascii=False))
                    f.write(",\n")
                    # f.write(json.dumps('measure_result = \n', ensure_ascii=False))
                    # f.write(json.dumps(measure_result, ensure_ascii=False))
                    # f.write(",\n")
                    f.write(json.dumps("accuracy：%.2f" % accuracy_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                    f.write(json.dumps("precision：%.2f" % precision_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                    f.write(json.dumps("recall：%.2f" % recall_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                    f.write(json.dumps("f1-score：%.2f" % f1_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                break
    torch.save(myModel.state_dict(), './combine_param.pth')


def test(datadir, num_n):
    # 数据加载
    data_f = "test.csv"
    loader = data_process.dataprocess(data_f, datadir)
    myModel.load_state_dict(torch.load('combine_param.pth'))
    myModel.eval()
    for epoch in range(1):
        acc = []
        loss_list = []
        label_true = []
        label_pred = []
        acc_all = 0
        loss_all = 0
        print("epoch数：" + str(epoch + 1))
        for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(loader):
            if (torch.cuda.is_available()):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device), labels.to(device)
            out = myModel(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)

            loss = criterion(out, labels)
            out = out.argmax(dim=1)
            label_pred = label_pred + out.tolist()
            label_true = label_true + labels.tolist()

            if i % 10 == 0:
                # print(out)
                accuracy = (out == labels).sum().item() / len(labels)
                #print(out)
                #print(labels)
                #label_pred = label_pred + out.tolist()
                #label_true = label_true + labels.tolist()
                acc.append(accuracy)
                loss_list.append(float(loss))
                print(i, loss.item(), accuracy)

            if i == 70:  # 2000条
                for num in range(len(acc)):
                    acc_all = acc_all + acc[num]
                    loss_all = loss_all + loss_list[num]
                acc_all = acc_all / len(acc)
                loss_all = loss_all / len(loss_list)
                print(loss_all, acc_all)
                measure_result = classification_report(label_true, label_pred)
                print('measure_result = \n', measure_result)
                print("accuracy：%.4f" % accuracy_score(label_true, label_pred))
                print("precision：%.4f" % precision_score(label_true, label_pred))
                print("recall：%.4f" % recall_score(label_true, label_pred))
                print("f1-score：%.4f" % f1_score(label_true, label_pred))
                acc_list.append(accuracy_score(label_true, label_pred))
                loss_list.append(loss_all)
                precision_list.append(precision_score(label_true, label_pred))
                recall_list.append(recall_score(label_true, label_pred))
                f1_list.append(f1_score(label_true, label_pred))
                with open(result_f, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(num_n, ensure_ascii=False))
                    f.write(",\n")
                    f.write(json.dumps("accuracy：%.2f" % accuracy_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                    f.write(json.dumps("precision：%.2f" % precision_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                    f.write(json.dumps("recall：%.2f" % recall_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                    f.write(json.dumps("f1-score：%.2f" % f1_score(label_true, label_pred), ensure_ascii=False))
                    f.write(",\n")
                break

for process in range(10):
    datadir = r"/root/autodl-tmp/data/实验_plus/N折交叉验证/ndata_" + str(process+1)
    train(datadir)
    print("test:")
    test(datadir, process+1)
acc_total = 0
loss_total = 0
pre_total = 0
recall_total = 0
f1_total = 0
for num in range(len(acc_list)):
    acc_total += acc_list[num]
    loss_total += loss_list[num]
    pre_total += precision_list[num]
    recall_total += recall_list[num]
    f1_total += f1_list[num]
print("end:")
print("acc:" + str(acc_total/10) + " | " +
      "loss:" + str(loss_total/10) + " | " +
      "precision:" + str(pre_total/10) + " | " +
      "recall:" + str(recall_total/10) + " | " +
      "f1:" + str(f1_total/10))

#torch.save(myModel.state_dict(), './combine_param.pth')
