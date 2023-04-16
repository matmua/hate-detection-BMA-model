import torch
import data_process
import model
from transformers import AdamW
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#bert测试
def berttest():
    # 模型试算
    pretrained = model.pretrained
    out = pretrained(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids)
    print("out.last_hidden_state.shape:")
    print(out.last_hidden_state.shape)


#数据加载
data_f = "litdata2.csv"
loader = data_process.dataprocess(data_f)
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break
input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)


berttest()

meanmodel = model.LSTM_attention()
meanmodel.load_state_dict(torch.load('Sentiment_param.pth'))
meanmodel.to(device)

'''print(meanmodel(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)'''


#训练
optimizer = AdamW(meanmodel.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
acc = []
loss_list = []
acc_all = 0
loss_all = 0
meanmodel.eval()
for epoch in range(1):
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        #print(input_ids)
        #print(labels)
        if (torch.cuda.is_available()):
            input_ids, attention_mask, token_type_ids,labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device),labels.to(device)
        out, feature = meanmodel(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        #print(out)
        loss = criterion(out, labels)
        ''' loss.backward()
        optimizer.step()
        optimizer.zero_grad()'''

        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        acc.append(accuracy)
        loss_list.append(float(loss))
        #print(acc)
        #print(loss_list)
        if i % 10 == 0:
            print(i, loss.item(), accuracy)

        if i == 124:#2000条
            break
#print(len(acc))
for num in range(len(acc)):
    acc_all = acc_all + acc[num]
    loss_all = loss_all + loss_list[num]
    #print(acc_all, loss_all)

acc_all = acc_all / len(acc)
loss_all = loss_all / len(loss_list)
print(loss_all, acc_all)


#torch.save(meanmodel.state_dict(), './hate_param.pth')
#mean_model = 0.3329676606655121 0.857
#sentiment_model = 0.43994053196907046 0.829 -----2.0955667514801024 0.342
