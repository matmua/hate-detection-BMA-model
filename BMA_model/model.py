from __future__ import unicode_literals, print_function, division
import torch
from torch import nn
import numpy as np
from transformers import BertModel, BertTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
"""
模型部分
"""

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

#加载预训练模型
pretrained = BertModel.from_pretrained('/root/autodl-tmp/Bert_Lstm-main/Bert_Lstm-main/bert-base-chinese')
pretrained.to(device)
embedding = pretrained.embeddings
# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

class bert:
    """
    需要用到的bert相关组件
    """
    def __init__(self, config):
        """
        初始化
        Args:
            config: 实例化的参数管理器
        """
        self.config = config
        #self.bert = BertModel.from_pretrained(self.config.bert_path)
        #self.bert = BertModel.from_pretrained('chinese-bert-wwm-ext')
        self.bert = BertModel.from_pretrained('/root/autodl-tmp/Bert_Lstm-main/SoftMaskedBert-main/model/chinese-bert-wwm-ext')
        # 加载预训练的模型
        self.embedding = self.bert.embeddings # 实例化BertEmbeddings类
        self.bert_encoder = self.bert.encoder
        # 实例化BertEncoder类，即attention结构，默认num_hidden_layers=12，也可以去本地bert模型的config.json文件里修改
        # 论文里也是12，实际运用时有需要再改
        # 查了源码，BertModel这个类还有BertEmbeddings、BertEncoder、BertPooler属性，在此之前我想获得bert embeddings都是直接用BertModel的call方法的，学习了
        self.tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/Bert_Lstm-main/SoftMaskedBert-main/model/chinese-bert-wwm-ext')  # 加载tokenizer
        self.masked_e = self.embedding(torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long))
        # 加载[mask]字符对应的编码，并计算其embedding
        self.vocab_size = self.tokenizer.vocab_size  # 词汇量

class biGruDetector(nn.Module):#错别字存在检测
    """
    论文中的检测器
    """
    def __init__(self, input_size, hidden_size, num_layer=1):
        """
        类初始化
        Args:
            input_size: embedding维度
            hidden_size: gru的隐层维度
            num_layer: gru层数
        """
        super(biGruDetector, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layer,
                          bidirectional=True, batch_first=True)
        # GRU层
        self.linear = nn.Linear(hidden_size * 2, 1)
        # 线性层
        # 因为双向GRU，所以输入维度是hidden_size*2；因为只需要输出个概率，所以第二个维度是1

    def forward(self, inp):
        """
        类call方法的覆盖
        Args:
            inp: 输入数据，embedding之后的！形如[batch_size,sequence_length,embedding_size]

        Returns:
            模型输出
        """
        rnn_output, _ = self.rnn(inp)
        # rnn输出output和最后的hidden state，这里只需要output；
        # 在batch_first设为True时，shape为（batch_size,sequence_length,2*hidden_size）;
        # 因为是双向的，所以最后一个维度是2*hidden_size。
        #print(rnn_output.shape)
        output = nn.Sigmoid()(self.linear(rnn_output))
        # sigmoid函数，没啥好说的，论文里就是这个结构
        #print(output.shape)
        #可对output降维处理
        return output, rnn_output
        # output维度是[batch_size, sequence_length, 1]

class LSTM_mood(nn.Module):
    def __init__(self, **kwargs):
        super(LSTM_mood, self).__init__()
        self.hidden_dim = 384  # 隐藏层节点数
        self.num_layers = 2  # 神经元层数
        self.n_class = 3  # 类别数

        self.bidirectional = False  # 控制是否双向LSTM
        # LSTM
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=0.5, batch_first=True)

        # weiht_w即为公式中的h_s(参考系)
        # nn. Parameter的作用是参数是需要梯度的
        self.weight_W = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 1))

        # 对weight_W、weight_proj进行初始化
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

        if self.bidirectional:
            self.fin_feature = nn.Linear(self.hidden_dim * 2, 64)
            self.fc = nn.Linear(64, self.n_class)
        else:
            self.fin_feature = nn.Linear(self.hidden_dim, 64)
            self.fc = nn.Linear(64, self.n_class)

        '''if self.bidirectional:
            self.fc = nn.Linear(self.hidden_dim * 2, self.n_class)
        else:
            self.fc = nn.Linear(self.hidden_dim, self.n_class)'''

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        前向传播
        :param inputs: [batch, seq_len]
        :return:
        """
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        out = out.last_hidden_state[:, 0]
        #print(out)
        # 编码
        #embeddings = self.embedding(out)  # [batch, seq_len] => [batch, seq_len, embed_dim][64,65,50]
        # Set initial hidden and cell states
        #print("outsize:" + str(out.size(0)))
        h0 = torch.zeros(2, out.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, out.size(0), self.hidden_dim).to(device)
        #print("c0:" + str(c0.shape))
        # 经过LSTM得到输出，state是一个输出序列
        # 结合batch_first设置
        out = out.unsqueeze(2).permute(0, 2, 1)
        #print("out:" + str(out.shape))
        states, (hidden_last, cn_last) = self.lstm(out,(h0, c0))  # [batch, seq_len, embed_dim]

        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]
        ''' 
        # print("states.shape=", states.shape)  (64,50,200)
        # attention
        # states与self.weight_W矩阵相乘，然后做tanh
        u = torch.tanh(torch.matmul(states, self.weight_W))
        # u与self.weight_proj矩阵相乘,得到score
        att = torch.matmul(u, self.weight_proj)
        # softmax
        att_score = F.softmax(att, dim=1)
        # 加权求和
        scored_x = states * att_score
        encoding = torch.sum(scored_x, dim=1)
        # 线性层
        print(states.shape)
        outputs = self.fc(states)
        '''
        #print(hidden_last_out.shape)
        feature = self.fin_feature(hidden_last_out)
        outputs = self.fc(feature)
        return outputs, feature

class softMaskedBert(nn.Module):
    """
    softmasked bert模型
    """
    def __init__(self, **kwargs):
        """
        类初始化
        Args:
            config: 实例化的参数管理器
        """
        super(softMaskedBert, self).__init__()
        #self.config = config  # 加载参数管理器
        self.vocab_size = kwargs['vocab_size']
        self.masked_e = kwargs['masked_e']
        self.bert_encoder = kwargs['bert_encoder']

        self.linear = nn.Linear(768, self.vocab_size)  # 线性层，没啥好说的
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # LogSoftmax就是对softmax取log

    def forward(self, bert_embedding, p, input_mask=None):
        """
        call方法
        Args:
            bert_embedding: 输入序列的bert_embedding
            p: 检测器的输出，表示输入序列对应位置的字符错误概率，维度：[batch_size, sequence_length, 1]
            input_mask: extended_attention_mask，不是单纯的输入序列的mask，具体使用方法见下面的代码注释
        Returns:
            模型输出，经过了softmax和log，维度[batch_size,sequence_length,num_vocabulary]
        """
        soft_bert_embedding = p * self.masked_e + (1 - p) * bert_embedding  # detector输出和[mask]的embedding加权求和
        bert_out = self.bert_encoder(hidden_states=soft_bert_embedding, attention_mask=input_mask)
        # 之后再经过一个encoder结构
        # 这里有个大坑，原本看transformer的手册，BertModel的attention mask是用于输入的遮挡，维度是[batch_size,sequence_length]，但是这么输入肯定报错，
        # 查源码得知，encoder使用这个参数的时候做了处理（主要是因为多头注意力机制），直接在encoder模块里传入最初的mask就会报错
        # 源代码备注：
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # 不能直接传入维度是[batch_size,sequence_length]的mask！具体处理方案见train.py
        h = bert_out[0] + bert_embedding  # 残差
        #print(h.shape)
        out = self.log_softmax(self.linear(h))  # 线性层，再softmax输出
        # out维度：[batch_size,sequence_length,num_vocabulary]
        #若要使用，需要对out的特征进行再次提取，降低维度
        return out, h

class MeanModel(nn.Module):
    def __init__(self):
        super(MeanModel, self).__init__()
        self.linear_sentence = nn.Linear(768, 64)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.linear_end = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        bert_feature = self.linear_sentence(out.last_hidden_state[:, 0])
        #print(bert_feature1.shape)
        bert_feature = self.relu(bert_feature)
        bert_feature = self.dropout(bert_feature)
        out = self.linear_end(bert_feature)
        #bert_feature = [16*64]
        return out, bert_feature


class LSTM_attention(nn.Module):
    def __init__(self, **kwargs):
        super(LSTM_attention, self).__init__()
        self.hidden_dim = 384  # 隐藏层节点数
        self.num_layers = 2  # 神经元层数
        self.n_class = 3  # 类别数

        self.bidirectional = True  # 控制是否双向LSTM
        # LSTM
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=0.5, batch_first=True)

        # weiht_w即为公式中的h_s(参考系)
        # nn. Parameter的作用是参数是需要梯度的
        self.weight_W = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 1))

        # 对weight_W、weight_proj进行初始化
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        if self.bidirectional:
            self.fin_feature = nn.Linear(self.hidden_dim * 2, 64)
            self.fc = nn.Linear(64, self.n_class)
        else:
            self.fin_feature = nn.Linear(self.hidden_dim, 64)
            self.fc = nn.Linear(64, self.n_class)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        前向传播
        :param inputs: [batch, seq_len]
        :return:
        """
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = out.last_hidden_state[:, 0]
        # 编码
        #embeddings = self.embedding(out)  # [batch, seq_len] => [batch, seq_len, embed_dim][64,65,50]
        # Set initial hidden and cell states
        #print("outsize:" + str(out.size(0)))
        h0 = torch.zeros(2*2, out.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2*2, out.size(0), self.hidden_dim).to(device)
        #print("c0:" + str(c0.shape))
        # 经过LSTM得到输出，state是一个输出序列
        # 结合batch_first设置
        out = out.unsqueeze(2).permute(0, 2, 1)
        #print("out:" + str(out.shape))
        states, (hidden_last, cn_last) = self.lstm(out,(h0, c0))  # [batch, seq_len, embed_dim]

        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]
        ''' 
        # print("states.shape=", states.shape)  (64,50,200)
        # attention
        # states与self.weight_W矩阵相乘，然后做tanh
        u = torch.tanh(torch.matmul(states, self.weight_W))
        # u与self.weight_proj矩阵相乘,得到score
        att = torch.matmul(u, self.weight_proj)
        # softmax
        att_score = F.softmax(att, dim=1)
        # 加权求和
        scored_x = states * att_score
        encoding = torch.sum(scored_x, dim=1)
        # 线性层
        print(states.shape)
        outputs = self.fc(states)
        '''
        #hidden_last_out = [16*768]
        feature = self.fin_feature(hidden_last_out)
        outputs = self.fc(feature)
        return outputs, feature


class Classifier(nn.Module):
    def __init__(self, in_dim=128):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.linear1 = nn.Linear(192, 2)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(96, 64)
        self.linear4 = nn.Linear(128, 2)
        self.linear5 = nn.Linear(64, 2)
        self.linear6 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, mean_feature, metaphor_feature, type):
        #compare_feature = self.linear3(compare_feature)
        # 1 : bert + gcn + compare
        # 2 : bert + gcn
        # 3 : bert + compare
        # 4 : gcn + compare
        if type == 0:
            feature = torch.cat((mean_feature, metaphor_feature), dim=1)
            # print(type(feature), feature.shape, feature.dtype)
            x = self.linear1(feature)
            x = self.linear2(x)
        if type == 1:
            metaphor_feature = self.linear6(metaphor_feature)
            feature = torch.cat((mean_feature, metaphor_feature), dim=1)
            x = self.linear3(feature)
            x = self.relu(x)
            x = self.linear5(x)
            #print(mean_feature)
            #print(metaphor_feature)
            #x = self.linear2(x)
        '''if type == 1:
            feature = torch.cat((bert_feature, gcn_feature, compare_feature), dim=1)
            # print(type(feature), feature.shape, feature.dtype)
            x = self.linear1(feature)
            x = self.linear2(x)
        elif type == 2:
            feature = torch.cat((bert_feature, gcn_feature), dim=1)
            x = self.linear4(feature)
        elif type == 3:
            feature = torch.cat((bert_feature, compare_feature), dim=1)
            x = self.linear4(feature)
        elif type == 4:
            feature = torch.cat((gcn_feature, gcn_feature), dim=1)
            x = self.linear4(feature)
        elif type == 5:
            x = self.linear5(bert_feature)
        elif type == 6:
            x = self.linear5(gcn_feature)
        elif type == 7:
            x = self.linear5(compare_feature)'''

        x = self.relu(x)
        x = self.dropout(x)
        x = F.log_softmax(x, dim=1)
        return x


class CombineModel(nn.Module):
    def __init__(self, bert_dim=768, gcn_dim=256):
        super(CombineModel, self).__init__()
        #self.BertModel = BertModel()
        # self.LstmRNN = LstmRNN(384, 20, output_size=64, num_layers=1)
        self.MeanModel = MeanModel()
        self.MeanModel.load_state_dict(torch.load('mean_param.pth'))
        self.LSTM_attention = LSTM_attention()
        self.LSTM_attention.load_state_dict(torch.load('Sentiment_param.pth'))
        self.MoodModel = LSTM_mood()
        self.MoodModel.load_state_dict(torch.load('mood_param.pth'))
        self.biGruDetector = biGruDetector(768, 50)
        self.biGruDetector.load_state_dict(torch.load('Detect_param.pth'))
        self.linear1 = nn.Linear(192, 2)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(128, 2)
        self.linear5 = nn.Linear(64, 2)
        self.linear6 = nn.Linear(64, 64)
        self.linear7 = nn.Linear(64, 64)
        self.linear8 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        for param in self.MeanModel.parameters():
            param.requires_grad_(True)
        for param in self.LSTM_attention.parameters():
           param.requires_grad_(False)
        #self.Classifier = Classifier(in_dim=128)
        for param in self.biGruDetector.parameters():
            param.requires_grad_(False)
        for param in self.MoodModel.parameters():
            param.requires_grad_(False)
        #self.SoftMask_model = softMaskedBert()


    def forward(self, input_ids, attention_mask, token_type_ids):

        #bert_feature = self.BertModel(bert_feature)
        # lstm_model = LstmRNN(6844, 20, output_size=64, num_layers=1)  # 20 hidden units
        # bert_feature = self.LstmRNN(bert_feature_1).to(device)
        mean_out, mean_feature = self.MeanModel(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #gcn_feature = self.GcnModel(gcn_feature)
        meta_out, metaphor_feature = self.LSTM_attention(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mood_out, mood_feature = self.MoodModel(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #knowledge_feature = self.Compare_Model(knowledge_feature)
        batch_inp_embedding = embedding(input_ids).to(device)
        faultdetect_feature, faultdetect_detail = self.biGruDetector(batch_inp_embedding)
        faultdetect_feature = faultdetect_feature.squeeze()
        if faultdetect_feature.shape[1] >=64:
            faultdetect_feature = torch.hsplit(faultdetect_feature, [64])[0]
        else:
            tensor_cloth = torch.zeros(8, 64-faultdetect_feature.shape[1]).to(device)
            faultdetect_feature = torch.cat((faultdetect_feature, tensor_cloth), dim=1)

        '''feature_list = []
        for i in tweet_type:
            feature_list.append(knowledge_feature[i-1])
        feature_list = torch.stack(feature_list, 0)

        compare_feature = torch.cat(((bert_feature - feature_list), torch.mul(bert_feature, feature_list)), 1)
        # compare_feature = torch.cat((feature_list, feature_list), 1)'''
        '''print("mean:")
        print(mean_feature)
        print("metaphor_feature:")
        print(metaphor_feature)
        print("faultdetect_feature:")
        print(faultdetect_feature)'''
        metaphor_feature = self.linear6(metaphor_feature)
        faultdetect_feature = self.linear7(faultdetect_feature)
        feature = torch.cat((mean_feature, metaphor_feature, faultdetect_feature, mood_feature), dim=1)
        x = self.linear8(feature)
        #result = self.Classifier(mean_feature, metaphor_feature, type=1)
        #x = self.linear5(mean_feature)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = F.log_softmax(x, dim=1)
        # 1 : bert + gcn + compare
        # 2 : bert + gcn
        # 3 : bert + compare
        # 4 : gcn + compare

        return x