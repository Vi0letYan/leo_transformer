import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用 yaml 配置文件
from utils.config import config

# 从配置文件中获取 GPU 信息
DEVICE = config.device


'''
创建一个 clones 函数，用于克隆模型块
module: 要克隆的网络结果
N: 克隆的数量
'''
def clones(module, N):
    '''
    克隆模型块，克隆的模型块参数不共享
    module: 要克隆的网络结果
    N: 克隆的数量
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

'''
创建 embedding 层
将输入的 被tokenizer分词后的离散词索引（词表索引） 转换为连续的向量表示
embedding 层是针对整个词表的，传入的是整个被tokenizer分词后的词表的尺寸大小，也即词表的子词总数
索引 -> 向量
'''
class Embeddings(nn.Module):
    # 创建 embedding 层时的初始化，需要传入 token 被向量化的维度，以及词表的尺寸大小
    # d_token_embedding: token 被向量化的维度
    # vocab_size: 被tokenizer分词后的词表的尺寸大小，也即词表的子词总数
    def __init__(self, d_token_embedding, vocab_size):
        super(Embeddings, self).__init__()
        '''
        创建 embedding 层，将词表的尺寸大小映射为 d_token_embedding 维的向量
        词嵌入矩阵 nn.Embedding 实际上创建了一个 词表的尺寸大小 × 被向量化的维度 的矩阵，用于将词表的尺寸大小映射为 d_token_embedding 维的向量
        同时，这个矩阵是可训练的，也即在训练过程中，这个矩阵的参数会被优化，使得模型能够更好地学习到词表的语义信息
        等价于 self.weight = torch.randn(vocab_size, d_token_embedding)  # 随机初始化
        等效于一个没有偏置的全连接层，输入维度为词表索引味道，输出为 d_token_embedding 维的向量
        但是全连接层是矩阵乘法，而词嵌入矩阵是直接索引（因为输入的词表索引本身就是独热编码，vocab_size维度中只有一个1，其余为0），因此词嵌入矩阵的计算效率更高
        '''
        self.lut = nn.Embedding(vocab_size, d_token_embedding)
        # 存储 token 被向量化的维度，用于后续可能的计算
        self.d_token_embedding = d_token_embedding
    
    def forward(self, input_tokens):
        '''
        前向传播函数，将输入的 被tokenizer分词后的离散词索引（词表索引） 转换为连续的向量表示
        input_tokens: 被tokenizer分词后的离散词索引（词表索引）
        return: 连续的向量表示
        最后返回向量乘以 math.sqrt(self.d_token_embedding) 适当增加词向量的方差，使与位置编码相加后，词向量处于主导位置
        '''
        return self.lut(input_tokens) * math.sqrt(self.d_token_embedding)

'''
为每个 token 的每个元素添加一个唯一的位置编码
这个编码会被添加到词向量中，使模型能够理解位置信息
输出：位置编码 + 词向量
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_token_embedding, dropout, max_tokens_length=5000):
        '''
        初始化位置编码
        d_token_embedding: token 被向量化的维度
        dropout: dropout 概率
        max_len: 最大长度
        '''
        super(PositionalEncoding, self).__init__()
        # 创建 dropout 层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个 size 为 max_tokens_length × d_token_embedding 的全零矩阵
        pe = torch.zeros(max_tokens_length, d_token_embedding, device=DEVICE)

        # 根据公式计算每个 token 的每个元素的位置编码
        # 公式：PE(pos, 2i) = sin(pos / (10000^(2i/d_token_embedding)))
        # 公式：PE(pos, 2i+1) = cos(pos / (10000^(2i/d_token_embedding)))
        # pos: position index, 即每个 token 的索引
        # i: dimension index, 即每个 token 的每个元素的索引
        # d_token_embedding: token 被向量化的维度
        # unsqueeze(1) 在第 1 维度上增加一个维度，且 size 为 1
        # 原本的形状为 (max_tokens_length,)，增加后变为 (max_tokens_length, 1)
        position = torch.arange(0. , max_tokens_length, device=DEVICE).unsqueeze(1)
        # i 的取值范围为 0 到 d_token_embedding-1，步长为 2
        # i = 0，1 的时候 sin cos 内部的值是相同的，所以可以直接步长为 2 统一计算括号内部值
        div_term = torch.exp(torch.arange(0. , d_token_embedding, 2, device=DEVICE) * -(math.log(10000.0) / d_token_embedding))

        # 计算每个 token 的每个元素的位置编码
        # position: (max_tokens_length, 1)
        # div_term: (d_token_embedding/2,)
        # 维度不同会自动广播，所以可以直接相乘
        # position 有 max_tokens_length 行，1 列，所以复制 max_tokens_length 列
        # div_term 有 d_token_embedding/2 个单元素，先变为 1 行，d_token_embedding/2 列，再复制 max_tokens_length 行
        # 乘积结果为 (max_tokens_length, d_token_embedding/2)，然后对于偶数维度，使用 sin 函数，奇数维度使用 cos 函数
        # 存储到 pe 矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 在第 0 维度上增加一个维度，且 size 为 1，使得 pe 的形状变为 (1, max_tokens_length, d_token_embedding)
        pe = pe.unsqueeze(0)
        # 将 pe 矩阵以持久的 buffer 状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)
    
    def forward(self, embedded_tokens):
        '''
        前向传播函数，将输入的词向量与位置编码相加
        embedded_tokens: 词向量
        return: 词向量 + 位置编码，并进行 dropout 操作
        '''
        with torch.no_grad():
            # pe 的形状为 (1, max_tokens_length, d_token_embedding)，一次性计算所有位置的编码
            # 因为位置编码是固定的，所以不需要计算梯度
            # 取出与 embedded_tokens 的序列长度相同的编码，作为位置编码
            positional_encoding = self.pe[:, :embedded_tokens.size(1)]
        embedded_tokens_with_position = embedded_tokens + positional_encoding
        return self.dropout(embedded_tokens_with_position)

'''
实现注意力机制的核心矩阵计算，只是方法，不是网络层，所以单独定义
这里的输入是跟权重矩阵相乘后的结果
Q: (batch_size, n_heads, seq_len, d_k)
K: (batch_size, n_heads, seq_len, d_k)
V: (batch_size, n_heads, seq_len, d_k)
mask: (batch_size, n_heads, seq_len, seq_len)
return: (batch_size, n_heads, seq_len, d_K)
'''
def attention(Q, K, V, mask=None, dropout=None):
    # 读取 Q 矩阵最后一个维度的尺寸 d_k
    # Q.size(-1) 表示 Q 矩阵的最后一个维度的大小
    d_k = Q.size(-1)

    # 将 K 矩阵的最后两个维度转置，才能与 Q 矩阵相乘，乘完了还要除以 d_k 开根号
    # K 矩阵的最后两个维度为 (seq_len, d_k)，转置后为 (d_k, seq_len)
    '''
    torch.matmul 的一般规则：
    1. 如果两个张量都是1D：向量点积
    2. 如果两个张量都是2D：标准矩阵乘法
    3. 如果一个是1D，一个是2D：向量-矩阵乘法
    4. 如果都是高维（≥3D）：批量矩阵乘法
    '''
    # attetnion_scores: (batch_size, n_heads, seq_len, seq_len)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    '''
    这里的mask同样是矩阵，并且会有不同的类型
    Encoder：使用 padding mask 处理不等长序列
    Decoder：同时使用 padding mask 和 sequence mask
    padding mask 处理填充部分
    sequence mask 防止看到未来信息
    '''

    if mask is not None:
        # 将 mask 中为 0 的部分替换为 -1e9
        attention_scores = attention_scores.masked_fill(mask==0, -1e9)

    # 将 attention_scores 矩阵按最后一个维度进行 softmax
    # 对于某个 batch 的某个 head 的某个 token(q_i) 里的所有 k_j 进行 softmax
    # 就是最后一个维度里的每个元素进行 softmax
    softmaxed_attention_scores = F.softmax(attention_scores, dim=-1)

    if dropout is not None:
        softmaxed_attention_scores = dropout(softmaxed_attention_scores)
    
    # 计算包含注意力信息的 value 矩阵
    AV = torch.matmul(softmaxed_attention_scores, V)
    # 同时返回注意力矩阵跟value的乘积，以及注意力矩阵
    return AV, softmaxed_attention_scores


'''
具体实现多头注意力机制
新建一个类是因为这里是网络层，需要继承 nn.Module 类，并实现 forward 方法
'''
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_token_embedding, d_k, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        # 每个 head 的维度
        self.d_k = d_k
        # head 数量
        self.num_heads = num_heads
        # 多头扩展因子，单头维度为 d_k，多头为 num_heads * d_k
        # 定义4个全连接函数，用于将输入的 query, key, value 矩阵转换为 Q K V 矩阵，以及多头注意力矩阵 concat 后的变换矩阵
        self.linears = clones(nn.Linear(d_token_embedding, num_heads * d_k), 4)
        # 初始化注意力矩阵
        self.attn = None
        # 创建 dropout 层
        # 提前定义 dropout 层，可以直接传入 dropout 参数，而不是在 forward 方法中每次都创建
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 在第 1 维度上增加一个维度，且 size 为 1
            # 原本的形状为 (batch_size, seq_len, seq_len)，增加后变为 (batch_size, 1, seq_len, seq_len)
            # 所有head共享同一个mask
            mask = mask.unsqueeze(1)

        # 读取 batch size
        batch_size = query.size(0)

        # 将输入的 query, key, value 矩阵转换为 Q K V 矩阵
        '''
        l(x) 表示第 l 个全连接函数，x 表示输入矩阵
        zip(self.linears, (query, key, value)) 表示将 self.linears 和 (query, key, value) 打包成一个元组
        for l, x in zip(self.linears, (query, key, value)) 表示将打包后的元组拆开，l 表示第 l 个全连接函数，x 表示输入矩阵

        .view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) 表示
        将输入矩阵(batch_size, seq_len, num_heads * d_k) 拆分为 (batch_size, seq_len, num_heads, d_k)
        然后通过 transpose(1, 2) 将第 1 和 2 维度互换，得到 (batch_size, num_heads, seq_len, d_k)
        '''
        Q, K, V = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        # 调用 attention 函数计算得到 num_heads 个注意力矩阵跟 value 的乘积，以及注意力矩阵
        AV, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        # 将 num_heads 个注意力矩阵 concat 起来
        # 再次将第 1 和 2 维度互换，得到 (batch_size, seq_len, num_heads, d_k)
        # 再通过 contiguous().view(batch_size, -1, self.num_heads * self.d_k) 将第 2 和 3 维度 concat 起来，得到 (batch_size, seq_len, num_heads * d_k)
        AV = AV.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 使用 self.linears 中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        # 将包含多头注意力信息的 AV 矩阵变换为 (batch_size, seq_len, d_token_embedding)
        B = self.linears[-1](AV)

        return B


'''
定义 层归一化 类
'''
class LayerNorm(nn.Module):
    def __init__(self, d_token_embedding, eps=1e-6):
        super(LayerNorm, self).__init__()

        # 初始化 α 为全 1，β 为全 0
        self.a_2 = nn.Parameter(torch.ones(d_token_embedding))
        self.b_2 = nn.Parameter(torch.zeros(d_token_embedding))
        # 平滑项，避免除以 0
        self.eps = eps

    def forward(self, x):
        # 神经网络的输出，x形状为 (batch_size, seq_len, d_token_embedding)
        # 在最后一个维度上计算均值和方差,即每个token单独进行归一化
        # keepdim=True 确保输出的维度与输入相同,保持住最后一个维度仍然存在
        # mean: (batch_size, seq_len, 1)
        mean = x.mean(-1, keepdim=True)
        # std: (batch_size, seq_len, 1)
        std = x.std(-1, keepdim=True)

        # Layer Norm公式: y = a * (x - mean) / sqrt(std^2 + eps) + b
        # a_2: (d_token_embedding,)
        # b_2: (d_token_embedding,)
        # a_2 和 b_2 是可学习的参数，eps 是避免除以 0 的平滑项
        y = self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2
        return y


'''
定义 FFN 前馈神经网络 类
'''
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_token_embedding, d_ffn, dropout=0.1):
        '''
        PositionwiseFeedForward 的初始化函数
        d_token_embedding: token 被向量化的维度
        d_ffn: 前馈神经网络中间层的维度
        '''
        super(PositionwiseFeedForward, self).__init__()
        # 创建两个全连接层，用于将输入的 token 向量转换为中间层的向量，然后再转换回 token 向量
        self.w_1 = nn.Linear(d_token_embedding, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_token_embedding)
        # 创建 dropout 层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 将输入的 token 向量转换为中间层的向量
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


'''
定义 SublayerConnection 类
实现一个完整的残差链接
'''
class SublayerConnection(nn.Module):
    def __init__(self, d_token_embedding, dropout=0.1):
        super(SublayerConnection, self).__init__()
        # 创建层归一化层
        self.norm = LayerNorm(d_token_embedding)
        # 创建 dropout 层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # sublayer: multi-head attention 或者 feed forward 层
        # sublayer -> dropout -> Layer Norm -> add -> return
        
        # 尝试不同顺序的残差连接
        # x = x + self.norm(self.dropout(sublayer(x))) # 我理解的对于原文的优化
        x = x + self.dropout(sublayer(self.norm(x))) # pre-norm，性能极大优于 post-norm
        # x = self.norm(x + self.dropout(sublayer(x))) # post-norm（原文顺序）
        return x

'''
定义 EncoderLayer 类
一个 EncoderLayer 包含一个多头自注意力层和一个 FFN 前馈神经网络层
'''
class EncoderLayer(nn.Module):
    def __init__(self, d_token_embedding, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # 创建多头自注意力层
        self.self_attn = self_attn
        # 创建 ffn
        self.feed_forward = feed_forward
        # 串联多头自注意力层和ffn
        self.sublayer = clones(SublayerConnection(d_token_embedding, dropout), 2)
        # 存储 token 被向量化的维度
        self.d_token_embedding = d_token_embedding
    
    def forward(self, x, mask):
        # 将 embedding 层的输出进行多头自注意力计算
        # lambda x: self.self_attn(x, x, x, mask) 是一个匿名函数，用于将输入的多形式参数融合为一个单参数传递给多头自注意力层
        # 等价于 def sublayer(x): return self.self_attn(x, x, x, mask)
        # 因为 sublayer 的输入只能是单参数，所以需要将多个参数融合为一个单参数
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x

'''
定义 Encoder 类
一个 Encoder 包含多个 EncoderLayer
'''
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        # 创建一组共 num_layers 个独立的 EncoderLayer
        # self.layers 是一个列表，列表中包含 num_layers 个 EncoderLayer 实例
        self.layers = clones(encoder_layer, num_layers)
        # 创建层归一化层
        self.norm = LayerNorm(encoder_layer.d_token_embedding)

    def forward(self, x, mask):
        # 使用 for 循环连续输入 encoder_layer 层
        # 为什么只有一层 norm？ 后续可以尝试实验
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

'''
定义 DecoderLayer 类
一个 DecoderLayer 包含一个多头自注意力层、一个多头交叉注意力层和一个 FFN 前馈神经网络层
'''
class DecoderLayer(nn.Module):
    def __init__(self, d_token_embedding, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # 创建自注意力层
        self.self_attn = self_attn
        # 创建交叉注意力层
        self.cross_attn = cross_attn
        # 创建 ffn
        self.feed_forward = feed_forward
        # 创建串联实例
        self.sublayer = clones(SublayerConnection(d_token_embedding, dropout), 3)
        # 存储 token 被向量化的维度
        self.d_token_embedding = d_token_embedding
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x

'''
定义 Decoder 类
一个 Decoder 包含多个 DecoderLayer
'''
class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(Decoder, self).__init__()
        # 创建一组共 num_layers 个独立的 DecoderLayer 的列表实例
        self.layers = clones(decoder_layer, num_layers)
        # 创建层归一化层
        self.norm = LayerNorm(decoder_layer.d_token_embedding)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 后续尝试实验多层 norm
        for layer in self.layers:
            # 相同的 encoder_output （最后一层 encoder layer 的输出） 用于所有的 decoder layer
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

'''
定义 Generator 类
用于将 Decoder 的输出转换为词典大小的向量
'''
class Generator(nn.Module):
    def __init__(self, d_token_embedding, vocab_size):
        super(Generator, self).__init__()
        # 创建一个全连接层，用于将 Decoder 的输出转换为词典大小的向量
        self.proj = nn.Linear(d_token_embedding, vocab_size)

    def forward(self, x):
        # x 的尺寸为 (batch_size, seq_len, d_token_embedding)
        x = self.proj(x)
        return x
    
'''
定义 Transformer 类
一个 Transformer 包含一个 Encoder 和一个 Decoder，以及一个 Generator
构建完整的 Transformer 模型
'''
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.encoder(x, src_mask)
        return x
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        x = self.tgt_embed(tgt)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        # 传入 encoder 的输出作为 decoder 的输入
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        '''
        transformer 的输出截止到 decoder 的输出，不包含 generator 层
        不包含 generator 层的原因是：
        generator 的输出维度是 vocab_size 过于庞大
        直接包含在 transformer 中会导致显存占用过高
        后续采用手动 chunk 的方式，在 seq_len 维度上进行梯度累计，避免峰值显存占用过高
        '''
        return decoder_output


'''
定义 make_model 工厂函数
用于快速实例化 Transformer 模型
'''
def make_model(src_vocab_size, tgt_vocab_size, num_layers=6, d_token_embedding=512, d_k=64, d_ffn=2048, num_heads=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化 Attention 对象
    attn = MultiHeadedAttention(num_heads, d_token_embedding, d_k, dropout).to(DEVICE)
    # 实例化 FFN 对象
    ffn = PositionwiseFeedForward(d_token_embedding, d_ffn, dropout).to(DEVICE)
    # 实例化 Positional Encoding 对象
    position = PositionalEncoding(d_token_embedding, dropout).to(DEVICE)

    # 实例化 embedding 对象
    embeddings = Embeddings(d_token_embedding, src_vocab_size).to(DEVICE)
    src_embed = nn.Sequential(embeddings, c(position)).to(DEVICE)
    tgt_embed = nn.Sequential(embeddings, c(position)).to(DEVICE)
    
    # 实例化 Encoder_layer 对象
    encoder_layer = EncoderLayer(d_token_embedding, c(attn), c(ffn), dropout).to(DEVICE)
    # 实例化 Encoder 对象
    encoder = Encoder(encoder_layer, num_layers).to(DEVICE)

    # 实例化 Decoder_layer 对象
    decoder_layer = DecoderLayer(d_token_embedding, c(attn), c(attn), c(ffn), dropout).to(DEVICE)
    # 实例化 Decoder
    decoder = Decoder(decoder_layer, num_layers).to(DEVICE)


    # 实例化 Generator 对象
    generator = Generator(d_token_embedding, tgt_vocab_size).to(DEVICE)

    # 实例化 Transformer 对象
    transformer_model = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        generator
    ).to(DEVICE)

    # 初始化模型参数
    # 遍历模型中的所有参数
    for p in transformer_model.parameters():
        # 判断参数是否为二维或更高维（例如权重矩阵，而不是偏置向量）
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return transformer_model