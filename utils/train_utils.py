import torch
import torch.nn as nn

# 使用 yaml 配置文件
from utils.config import config

'''
创建Chunked Generator Loss Compute类，用于计算Chunked Generator的损失
generator: 生成器模型，通常是生成预测分布的网络。在当前的 transformer 中，generator 是 model.generator
criterion: 损失函数，用于计算预测和目标之间的差距。在当前的 transformer 中，criterion 是 torch.nn.CrossEntropyLoss
devives_list: 使用的GPU设备列表。在当前的 transformer 中，devives_list 是 config.device_id
opt: 优化器对象，进行参数更新。在当前的 transformer 中，opt 是 NoamOpt
chunk_size: 将数据分割成多个小块进行计算的大小
'''
class ChunkedGeneratorLossCompute:
    '''
    generator 是一个线性层，拥有巨大的参数量，输出维度为 vocab_size

    以一个简单的三层全连接层为例：

    显存占用始终不变：
    尺寸[input_dimension, linear_layer_output_dimension] 与输入的数据无关
    ├── W1.grad  [512, 2048]      
    ├── W2.grad  [2048, 2048]
    ├── W3.grad  [2048, 30000]    

    每个 chunk 临时存在：
    尺寸[batch_size, chunk_size, linear_layer_output_dimension] 与输入的数据有关
    ├── h1       [32, 10, 2048]   ← 前向的中间值
    ├── h2       [32, 10, 2048]
    ├── h3       [32, 10, 30000]
    ├── ∂L/∂h1   [32, 10, 2048]   ← 激活值梯度
    ├── ∂L/∂h2   [32, 10, 2048]   
    └── ∂L/∂h3   [32, 10, 30000]  

    手动且不断增加显存占用的：
    尺寸 [batch_size, chunk_size, d_token_embedding] -> [batch_size, seq_len, decoder_output_dimension]
    └── ∂L/∂x    [32, 10, 512]    ← 用于传回 decoder

    相当于在一个 batch 内手动进行 chunk 后梯度累计
    '''
    def __init__(self, generator, criterion, devives_list, opt=None, chunk_size=5):
        
        # 保存 generator 层
        self.generator = generator
        # 使用 nn.parallel.replicate 将损失函数复制到多个GPU
        self.criterion = nn.parallel.replicate(criterion, devices=devives_list) # 返回一个包含多个GPU的损失函数列表
        # 保存优化器对象
        self.opt = opt
        # 保存使用的GPU设备列表
        self.devives_list = devives_list
        # 保存将数据分割成多个小块进行计算的大小
        self.chunk_size = chunk_size

    def __call__(self, decoder_out, target_label, normalize):
        '''
        进行多GPU的损失计算和训练
        decoder_out: decoder 的输出
        target_label: 目标数据（真实标签）
        normalize: 用于规范化损失的常数
        return: 总损失值（乘以normalize）


        前向传播：
        输入: decoder_out (来自Transformer Decoder的输出)
                ↓
        nn.parallel.scatter() → 分发到多个GPU
                ↓
        按 chunk_size 切片 → chunked_decoder_output
                ↓
        .detach() → 切断与原计算图的连接
                ↓
        nn.parallel.parallel_apply(generator, ...) → 并行执行 Generator 层
                ↓
        chunked_generator_output (预测的词表概率分布)
                ↓
        展平为 [batch_size * chunk_size, vocab_size]
                ↓
        nn.parallel.parallel_apply(criterion, ...) → 并行计算 CrossEntropyLoss
                ↓
        nn.parallel.gather() → 汇总所有GPU的损失到主设备
                ↓
        l_.sum() / normalize → 归一化后的总损失


        反向传播：
        l_.backward()
                ↓
        梯度传递到 chunked_generator_output
                ↓
        梯度传递到 Generator 层的参数 (W.grad 更新)
                ↓
        梯度传递到 chunked_decoder_output
                ↓
        在 .detach() 处停止 ← 计算图被切断
                ↓
        手动保存: chunked_decoder_output.grad → chunked_model2decoder_grad

        torch.cat(chunked_model2decoder_grad) → 合并所有 chunk 的梯度
                ↓
        nn.parallel.gather() → 汇总到主设备
                ↓
        raw_decoder_output.backward(gradient=gatherd_model2decoder_grad)
                ↓
        梯度传递到 Decoder
                ↓
        梯度传递到 Encoder
                ↓
        opt.step() → 更新所有参数
                ↓
        opt.optimizer.zero_grad() → 清零梯度
        '''
        
        total = 0.0 # 初始化总损失
        # 将生成器复制到多个GPU上,返回一个包含多个GPU的生成器列表
        generator = nn.parallel.replicate(self.generator, devices=self.devives_list)
        # 将模型输出分发到多个GPU
        # decoder_out 的形状为 [batch_size, seq_len, d_token_embedding]
        # decoder_out 是 Transformer forward() 的返回值，它的计算图记录了整个前向过程
        decoder_out_scatter = nn.parallel.scatter(decoder_out, target_gpus=self.devives_list)
        # 初始化 chunked model 到 decoder 的梯度列表，用于手动补充被切断的计算图的梯度
        chunked_model2decoder_grad = [[] for _ in decoder_out_scatter]
        # 将目标标签分发到多个GPU
        target_label = nn.parallel.scatter(target_label, target_gpus=self.devives_list)
        
        # 将数据划分为多个块（chunk），进行批量计算
        '''
        序列长度 seq_len = 15, chunk_size = 5

        for循环:
        i=0  → 处理位置 [0:5]   → 所有GPU并行计算这个chunk
        i=5  → 处理位置 [5:10]  → 所有GPU并行计算这个chunk  
        i=10 → 处理位置 [10:15] → 所有GPU并行计算这个chunk
        '''
        chunk_size = self.chunk_size
        for i in range(0, decoder_out_scatter[0].size(1), chunk_size):
            '''
            o[:, i:i + chunk_size]：从每个GPU的输出 o (for o in decoder_out_scatter) 中切片，提取第 i 到 i + chunk_size 列的数据（按时间步/序列位置切分）
            detach()：将张量从计算图中分离，清除之前的梯度历史,不参与梯度计算
            requires_grad_(self.opt is not None)：如果优化器存在，则设置为需要梯度计算，否则不设置梯度计算
            '''
            # 将每个 GPU 的 decoder 输出进行切片
            chunked_decoder_output = [[o[:, i:i + chunk_size].detach().requires_grad_(self.opt is not None)]
                    for o in decoder_out_scatter] # 返回一个包含多个GPU的输出列表

            # 并行运行 generator 进行预测
            # 这里使用 nn.parallel.parallel_apply 是因为前面手动断开计算图，后续需要手动保存 loss 并 backward
            chunked_generator_output = nn.parallel.parallel_apply(generator, chunked_decoder_output)

            # 计算每个 chunk 的损失
            # 遍历每个 GPU 的 generator 输出和 decoder 标签
            # 初始化预测值和目标值的列表
            chunked_target_label_list = []

            # 遍历每个GPU上的生成器输出和对应的目标
            for g, t in zip(chunked_generator_output, target_label):
                # g: 当前GPU的生成器输出，形状 [batch_size, chunk_size, vocab_size]
                # t: 当前GPU的目标标签，形状 [batch_size, seq_len]
                
                # 1. 处理预测值
                # 确保内存连续
                g_contiguous = g.contiguous()
                # 获取词表大小（最后一维）
                vocab_size = g.size(-1)
                # 展平为 [batch_size * chunk_size, vocab_size]
                pred = g_contiguous.view(-1, vocab_size)
                
                # 2. 处理目标值
                # 取出当前chunk对应的目标片段，形状 [batch_size, chunk_size]
                target_label_chunk = t[:, i:i + chunk_size]
                # 确保内存连续
                target_label_contiguous = target_label_chunk.contiguous()
                # 展平为一维 [batch_size * chunk_size]
                target_label_flatten = target_label_contiguous.view(-1)
                
                # 3. 组成元组，加入列表
                chunked_target_label_list.append((pred, target_label_flatten))

            # chunked_target_label_list 现在是一个列表，每个元素是 (pred, target_label) 元组
            # 例如 2 个 GPU 时: 
            # [(pred_gpu0_chunk0, target_label_gpu0_chunk0), (pred_gpu1_chunk0, target_label_gpu1_chunk0), (pred_gpu0_chunk1, target_label_gpu0_chunk1), (pred_gpu1_chunk1, target_label_gpu1_chunk1)]
            # 并行计算每个 chunk 的损失
            chunked_model_loss = nn.parallel.parallel_apply(self.criterion, chunked_target_label_list)

            # 汇总损失并进行规范化
            '''
            nn.parallel.gather(loss, target_device=self.devices[0])
            将所有GPU的损失值合并到主设备上（通常是第一个GPU）
            '''
            l_ = nn.parallel.gather(chunked_model_loss, target_device=self.devives_list[0])
            # 归一化 loss，计算平均损失
            l_ = l_.sum() / normalize
            # 累加整个 batch 的损失
            total += l_.data

            # 反向传播损失到Transformer的输出
            # enumerate 是 Python 内置函数，用于在遍历可迭代对象时同时获取索引和元素。
            # 遍历 multi_gpu_loss 列表，获取索引 j 和对应的损失 l
            if self.opt is not None:
                # 从总损失 l_ 反向传播，这里的计算图会自动计算并传递到 chunked_decoder_output
                l_.backward()
                for j, l in enumerate(chunked_model_loss):
                    '''
                    o[:, i:i + chunk_size].detach().requires_grad_(self.opt is not None)] for o in decoder_out_scatter
                    计算图在 .detach() 切断了，梯度传递到 chunked_decoder_output 而停止
                    所以需要 backward(gradient=chunked_model2decoder_grad) 来补充 decoder_out_scatter 的梯度
                    '''
                    # 取出每个GPU上 chunked_decoder_output 的梯度并保存到 chunked_model2decoder_grad 列表中
                    chunked_model2decoder_grad[j].append(chunked_decoder_output[j][0].grad.data.clone())
            
        # 反向传播所有损失，通过Transformer进行参数更新
        if self.opt is not None:
            # 将每个GPU的不同 chunk 需要手动从 generator 回传给 decoder 的梯度合并
            model2decoder_grad = [torch.cat(og, dim=1) for og in chunked_model2decoder_grad]

            # 保存 decoder 的原始信息（用于后续backward）
            raw_decoder_output = decoder_out
            # 将 model2decoder_grad 合并在主 rank 上
            gatherd_model2decoder_grad = nn.parallel.gather(model2decoder_grad, target_device = self.devives_list[0])

            # 手动补充 decoder 的梯度，进行反向传播更新参数，整个 encoder decoder 的所有梯度自动传播完成
            raw_decoder_output.backward(gradient=gatherd_model2decoder_grad)
            # 更新参数
            self.opt.step()
            # 清除梯度
            self.opt.optimizer.zero_grad()
        return total * normalize


class NoamOpt:
    def __init__(self, d_token_embedding, factor, warmup, optimizer):
        '''
        初始化优化器包装类
        d_token_embedding: 词嵌入维度
        factor: 学习率因子
        warmup: 预热步数
        optimizer: 优化器
        '''
        self.optimizer = optimizer # 优化器
        self._step = 0 # 当前训练步数
        self.warmup = warmup # 预热步数
        self.factor = factor # 学习率因子
        self.d_token_embedding = d_token_embedding # 词嵌入维度
        self._rate = 0 # 当前学习率

    def step(self):
        '''
        更新优化器的参数和学习率
        '''
        self._step += 1
        # 计算当前学习率
        rate = self.rate()
        # 更新优化器中所有参数的学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate # 设置当前学习率
        self._rate = rate # 保存当前学习率
        # 执行一次优化步骤（更新参数）
        self.optimizer.step()

    def rate(self, step=None):
        '''
        根据当前步数计算学习率
        '''
        # 如果没有传入step，使用当前步数
        if step is None:
            step = self._step
        # 学习率计算公式：factor * (d_token_embedding ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
        current_rate = self.factor * (self.d_token_embedding ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return current_rate


def get_std_opt(model):
    '''
    创建并返回一个NoamOpt优化器，包含Adam优化器作为基础
    '''
    NoamOpt_obj = NoamOpt(model.src_embed[0].d_token_embedding, config.lr_factor, config.warmup_steps,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    return NoamOpt_obj