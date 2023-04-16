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

#数据加载
data_f = "data329.csv"
loader = data_process.dataprocess(data_f)
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break
input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)

#bert测试
def berttest():
    # 模型试算
    pretrained = model.pretrained
    out = pretrained(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids)
    print("out.last_hidden_state.shape:")
    print(out.last_hidden_state.shape)

berttest()

#combineModel = model.CombineModel()
#combineModel.load_state_dict(torch.load('hate_param.pth'))
#combineModel.to(device)
combinemodel = model.CombineModel()
myModel = combinemodel
#myModel.load_state_dict(torch.load('mean_param.pth'))
myModel.to(device)

for param in combinemodel.parameters():
    param.requires_grad_(True)
for param in model.pretrained.parameters():
    param.requires_grad_(False)
'''print(meanmodel(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)'''


#训练
#optimizer = AdamW(combineModel.parameters(), lr=5e-4)
optimizer = AdamW(myModel.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
commentf = './programming.txt'


myModel.train()
for epoch in range(10):
    acc = []
    loss_list = []
    label_true = []
    label_pred = []
    acc_all = 0
    loss_all = 0
    print("epoch数：" + str(epoch+1))
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        #print(input_ids)
        #print(labels)
        if (torch.cuda.is_available()):
            input_ids, attention_mask, token_type_ids,labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device),labels.to(device)
        out = myModel(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        #print(out)
        #print(feature.shape)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            #print(out)
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            #print(out)
            #print(labels)
            label_pred = label_pred + out.tolist()
            label_true = label_true + labels.tolist()
            acc.append(accuracy)
            loss_list.append(float(loss))
            print(i, loss.item(), accuracy)

        if i == 415:#2000条
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
                f.write(json.dumps(epoch+1, ensure_ascii=False))
                f.write(",\n")
                #f.write(json.dumps('measure_result = \n', ensure_ascii=False))
                #f.write(json.dumps(measure_result, ensure_ascii=False))
                #f.write(",\n")
                f.write(json.dumps("accuracy：%.2f" % accuracy_score(label_true, label_pred), ensure_ascii=False))
                f.write(",\n")
                f.write(json.dumps("recall：%.2f" % recall_score(label_true, label_pred), ensure_ascii=False))
                f.write(",\n")
                f.write(json.dumps("f1-score：%.2f" % f1_score(label_true, label_pred), ensure_ascii=False))
                f.write(",\n")
            break

torch.save(myModel.state_dict(), './combine_param.pth')
