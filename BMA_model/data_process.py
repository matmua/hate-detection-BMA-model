import torch
from datasets import load_dataset
from transformers import BertTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('/root/autodl-tmp/Bert_Lstm-main/Bert_Lstm-main/bert-base-chinese')
print("token:")
print(token)



#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, data_f, dir):
        #self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)
        self.dataset = load_dataset("csv", data_dir=dir,
                                    data_files=data_f, split=split)
        #"test_with_label.csv"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['Sentence']
        label = self.dataset[i]['Label']
        return text, label

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #print(sents[0:16],labels[0:16])
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)
    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    #print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels

def dataprocess(data_f, dir):
    dataset = Dataset('train', data_f, dir)
    print(len(dataset), dataset[0])
    # 数据加载器
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=16,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)
    print(len(loader))
    return loader
