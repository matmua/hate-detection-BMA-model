"""
@File : config.py
@Description: 参数管理器
@Author: bin.chen
@Contact: LebesgueIntigrade@163.com
@Time: 2021/8/30
@IDE: Pycharm Professional
@REFERENCE: None
"""
import os
import platform
import torch


class Config(object):
    """
    参数管理器
    """
    def __init__(self):
        if self.platform() == 'Linux':
            current_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        elif self.platform() == 'Windows':
            current_dir = '/'.join(os.path.abspath(__file__).split('\\\\')[:-2])
        # 获取绝对项目地址，分割符视操作系统而定
        else:
            raise 'Unknown System'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.inference_device = 'cuda'
        self.inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.bert_path = '/root/autodl-tmp/Bert_Lstm-main/Bert_Lstm-main/bert-base-chinese'
        self.vocab_file = '/root/autodl-tmp/Bert_Lstm-main/Bert_Lstm-main/bert-base-chinese/vocab.txt'
        # 预训练的模型文件地址，去transformers hugface官网下载即可
        self.hidden_size = 50  # 隐层维度
        self.embedding_size = 768  # embedding维度
        self.lr = 0.0001
        self.epoch = 20
        self.batch_size = 8

        self.datapath = '/root/autodl-tmp/Bert_Lstm-main/SoftMaskedBert-main/data/narts/merge.csv'

        self.checkpoints = 'root/autodl-tmp/Bert_Lstm-main/SoftMaskedBert-main/model/checkpoints'

    @staticmethod
    def platform():
        """
        Returns:
            返回当前操作系统名称,'Linux'或'Windows'
        """
        return platform.system()





















