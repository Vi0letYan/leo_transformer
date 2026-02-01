import torch
import numpy as np
import json

from utils.config import config
from utils.tokenizer_loader import chinese_tokenizer_load, english_tokenizer_load
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

DEVICE = config.device


# 创建标准的 时序掩码 方法
def subsequent_mask(size):
    '''
    训练时，防止decoder看到未来的token

    # np.ones((1, 5, 5)) 创建：
    [[[1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]]]

    # np.triu(..., k=1) 后：
    [[[0, 1, 1, 1, 1],   # 主对角线及以下都是0
    [0, 0, 1, 1, 1],   # 主对角线上方都是1
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]]]
    '''
    # 为了通用性，创建一个形状为 (1, size, size) 的矩阵
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个从numpy转换为tensor的形状为 (1, size, size) 的矩阵，矩阵中为0的部分为True，为1的部分为False
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    '''
    该类用于存储一个训练批次的数据，包括源语言和目标语言的文本、token以及mask。
    所有的 mask 在数据准备阶段就被创建好了，在训练过程中直接使用。避免在训练过程中重复创建 mask，提高效率。
    '''
    def __init__(self, src_text, tgt_text, src, tgt=None, pad=0):
        '''
        src_text: 源语言（英语）的文本
        tgt_text: 目标语言（中文）的文本
        src: 源语言的输入数据（tensor格式）
        tgt: 目标语言的输入数据（tensor格式），如果有
        pad: padding值（用于填充句子时的标记）
        '''

        self.src_text = src_text  # 源语言文本
        self.tgt_text = tgt_text  # 目标语言文本
        src = src.to(DEVICE)   # 将源语言数据移到指定设备（GPU）
        self.src = src  # 保存在GPU上的源语言数据
        # 对于当前输入的句子非空部分进行判断成 bool 序列
        # 并在seq_length前面增加一维 ( -2 表示在倒数第二维增加一维)
        # ——> 维度为 batch_size × 1 × seq_length 的矩阵
        # (src != pad) 进行逐元素比较，不等于 pad 的部分为 True，等于 pad 的部分为 False
        self.src_mask = (src != pad).unsqueeze(-2)  # 生成源语言的mask，屏蔽padding部分

        # 处理 decoder 部分
        '''
        # 批次包含3个句子，长度不同
        # PAD=0, BOS=1, EOS=2
        原始句子:
        句子1: "我爱你"     → [BOS, 我, 爱, 你, EOS]     → [1, 5, 8, 12, 2]
        句子2: "他很好"     → [BOS, 他, 很, 好, EOS]     → [1, 3, 7, 9, 2] 
        句子3: "谢谢"       → [BOS, 谢, 谢, EOS]         → [1, 6, 6, 2]

        # 填充到相同长度（最长句子长度=5）
        tgt = torch.tensor([
            [1, 5, 8, 12, 2],    # 句子1：无需填充
            [1, 3, 7, 9, 2],     # 句子2：无需填充  
            [1, 6, 6, 2, 0]      # 句子3：末尾填充1个PAD
        ])

        
        步骤1：创建 decoder 输入 (self.tgt)

        self.tgt = tgt[:, :-1]  # 去掉每个句子的最后一个token

        结果:
        self.tgt = torch.tensor([
            [1, 5, 8, 12],       # [BOS, 我, 爱, 你]
            [1, 3, 7, 9],        # [BOS, 他, 很, 好]
            [1, 6, 6, 2]         # [BOS, 谢, 谢, EOS] ← 注意这里！
        ])


        步骤2：创建预测目标 (self.tgt_y)

        self.tgt_y = tgt[:, 1:]  # 从第二个token开始

        结果:
        self.tgt_y = torch.tensor([
            [5, 8, 12, 2],       # [我, 爱, 你, EOS]
            [3, 7, 9, 2],        # [他, 很, 好, EOS]
            [6, 6, 2, 0]         # [谢, 谢, EOS, PAD] ← PAD作为目标！
        ])


        步骤3：统计有效词数 (self.ntokens)

        self.ntokens = (self.tgt_y != pad).data.sum()

        计算过程:
        句子1: [5, 8, 12, 2] != 0 → [True, True, True, True]   → 4个词
        句子2: [3, 7, 9, 2]  != 0 → [True, True, True, True]   → 4个词  
        句子3: [6, 6, 2, 0]  != 0 → [True, True, True, False]  → 3个词

        总计: 4 + 4 + 3 = 11个有效词


        训练流程：
        批次中每个句子的训练过程:

        句子1 (长度5):
        输入位置:  [BOS] [我]  [爱]  [你]
                ↓     ↓     ↓     ↓
        预测目标:  [我]  [爱]  [你]  [EOS]
        损失计算:  ✓     ✓     ✓     ✓     (4个位置都计算损失)

        句子2 (长度5):  
        输入位置:  [BOS] [他]  [很]  [好]
                ↓     ↓     ↓     ↓
        预测目标:  [他]  [很]  [好]  [EOS]
        损失计算:  ✓     ✓     ✓     ✓     (4个位置都计算损失)

        句子3 (长度4，有padding):
        输入位置:  [BOS] [谢]  [谢]  [EOS]
                ↓     ↓     ↓     ↓
        预测目标:  [谢]  [谢]  [EOS] [PAD]
        损失计算:  ✓     ✓     ✓     ✗     (PAD位置不计算损失)
        '''
        if tgt is not None:
            tgt = tgt.to(DEVICE)  # 将目标语言数据移到指定设备
            # 创建 decoder 输入
            # 输入给 decoder 的句子，去掉最后一个 token
            self.decoder_input = tgt[:, :-1]
            
            # 创建 label 输出
            # 总体输出序列中，去掉第一个 token 后的目标序列，用于计算 loss 的 label
            self.target_label = tgt[:, 1:]

            # 创建 decoder 的输入 mask， 包括 padding mask 和 sequence mask
            self.decoder_input_mask = self.make_decoder_input_mask(self.decoder_input, pad)

            # 统计有效词数
            self.num_tokens = (self.target_label != pad).data.sum()
    
    # 标注Mask掩码，包括 padding mask 和 sequence mask
    '''
    # 普通方法（需要实例）
    class Example:
        def normal_method(self):
            pass

    # 调用方式
    obj = Example()
    obj.normal_method()  # ✓ 需要创建实例

    # 静态方法（不需要实例）
    class Example:
        @staticmethod
        def static_method():
            pass

    # 调用方式
    Example.static_method()  # ✓ 直接通过类调用
    obj = Example()
    obj.static_method()     # ✓ 也可以通过实例调用
    '''
    # 为了代码的简洁性，将创建 decoder 输入 mask 和输出 mask 的方法封装成一个静态方法
    # 同时，这是在 batch 创建时的代码流程，所以包含在 Batch 类中，而不是全局函数
    @staticmethod
    def make_decoder_input_mask(tgt, pad):
        '''
        tgt: 目标语言的 token 序列
        pad: padding 标记
        return: 目标语言的 mask，包括 padding mask 和 sequence mask
        '''
        # 添加 padding mask，tgt 中非 pad 的元素为 True，pad 的元素为 False
        tgt_padding_mask = (tgt != pad).unsqueeze(-2)   # 为目标语言中的非 pad 部分生成 mask
        
        # 添加 subsequent mask，防止 decoder 看到未来的 token
        # sequence mask 的输入是 tgt 的序列长度，即 tgt.size(-1)
        tgt_sequence_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_padding_mask.data)
        
        # 同时合并 padding mask 和 sequence mask
        tgt_mask = tgt_padding_mask & tgt_sequence_mask
        return tgt_mask  # 返回目标语言的 mask


'''
数据处理
创建自己的数据集类，继承自 PyTorch 的 Dataset 类
主要功能：加载英文和中文句子，使用分词器进行分词，将句子转换为token ID，并填充句子长度。
这里的 token ID 是 SentencePiece 分词器生成的。还没有被 Embedding 层转换为向量。是单值的整数。
'''
class MachineTranslationDataset(Dataset):
    def __init__(self, data_path):
        # 加载数据集和分词器
        self.raw_eng_sents, self.raw_zh_sents = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_zh = chinese_tokenizer_load()
        # 获取PAD、BOS、EOS标记ID, 用于填充和编码。特殊 token 的 ID 是固定且通用。
        self.PAD = self.sp_eng.pad_id()
        self.BOS = self.sp_eng.bos_id()
        self.EOS = self.sp_eng.eos_id()
    
    @staticmethod
    def len_argsort(seq):
        '''
        seq: 需要排序的句子（列表形式），每个元素是一个句子（字符串）
        return: 排序后的索引列表，索引对应的是原始句子列表的索引
        '''
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    
    def get_dataset(self, data_path, sort=False):
        '''
        data_path: 数据集文件路径（json格式）
        sort: 是否对数据按英文句子长度进行排序
        return: 英文句子列表和中文句子列表
        '''
        dataset = json.load(open(data_path, 'r', encoding="utf-8"))
        raw_eng_sents = []
        raw_zh_sents = []
        # 使用enumerate遍历数据集，获取索引和数据。用 _ 表示只使用索引，不使用数据。
        for idx, _ in enumerate(dataset):
            raw_eng_sents.append(dataset[idx][0])
            raw_zh_sents.append(dataset[idx][1])
        # 按照英文句子（源语言）长度进行排序
        # 因为模型在训练时，需要按照英文句子长度进行排序，以保证每个批次的数据长度相同，避免padding浪费计算资源。
        if sort:
            sorted_index = self.len_argsort(raw_eng_sents)
            raw_eng_sents = [raw_eng_sents[i] for i in sorted_index]
            raw_zh_sents = [raw_zh_sents[i] for i in sorted_index]
        return raw_eng_sents, raw_zh_sents

    '''
    PyTorch 的 Dataset 基类要求子类必须实现 __getitem__ 和 __len__ 方法。

    from torch.utils.data import DataLoader

    dataset = MTDataset("data.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # DataLoader 内部会调用：
    # - len(dataset) 来确定数据集大小
    # - dataset[idx] 来获取具体数据

    只有特定名称的方法才会被 Python 自动调用：
    - __getitem__ 对应 [] 操作符
    - __len__ 对应 len() 函数
    - __str__ 对应 print() 函数
    - __repr__ 对应 print() 函数
    - __add__ 对应 + 操作符
    - __radd__ 对应 + 操作符
    - __iadd__ 对应 += 操作符
    - __iter__ 对应 for 循环
    - __next__ 对应 next() 函数
    - __contains__ 对应 in 操作符
    - __getattr__ 对应 getattr() 函数
    - __setattr__ 对应 setattr() 函数
    - __delattr__ 对应 delattr() 函数
    - __getattribute__ 对应 getattr() 函数
    - __setattribute__ 对应 setattr() 函数
    - __delattribute__ 对应 delattr() 函数
    - __getitem__ 对应 [] 操作符
    - __setitem__ 对应 []= 操作符
    - __delitem__ 对应 del[] 操作符
    - __getslice__ 对应 [:] 操作符
    - __setslice__ 对应 [:] 操作符
    - __delslice__ 对应 [:] 操作符
    '''
    
    # __getitem__ 的命名方式，使对象支持索引访问，像列表一样使用 [] 操作符。
    def __getitem__(self, idx):
        '''
        idx: 数据索引
        return: 英文和中文句子对
        '''
        eng_text = self.raw_eng_sents[idx]
        zh_text = self.raw_zh_sents[idx]
        return [eng_text, zh_text]

    # __len__ 的命名方式，使对象支持 len() 函数，返回数据集大小。
    def __len__(self):
        '''
        return: 数据集大小
        '''
        return len(self.raw_eng_sents)

    def collate_fn(self, batch):
        '''
        对每个 batch 的样本，进行分词、填充、编码等操作。
        batch: 一个batch的样本
        return: 处理后的batch
        '''

        # 从batch中提取英文和中文文本
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        # 对英文和中文句子进行分词，并加上BOS和EOS标记
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_zh.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # 对英文和中文句子进行填充，保证每个句子的长度相同
        batch_src = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_tgt = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        # 返回一个Batch对象，包含源语言和目标语言的文本、token和mask
        batch = Batch(src_text, tgt_text, batch_src, batch_tgt, self.PAD)
        return batch