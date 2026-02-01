import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import sacrebleu
import logging
import warnings

from utils.beam_search import beam_search
from utils.create_exp_folder import create_exp_folder
from utils.data_loader import MachineTranslationDataset
from utils.config import config
from utils.train_utils import ChunkedGeneratorLossCompute, get_std_opt
from utils.tokenizer_loader import chinese_tokenizer_load
from model_codes.transformer_model import make_model


logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', level=logging.INFO)

def run_epoch(data, model, loss_compute):
    '''
    data: 数据集
    model: 模型
    loss_compute: 损失计算函数
    return: 每个token的平均损失

    Teacher Forcing 训练方式：
    预测下一个 token 时，使用真实目标前序 token 作为输入，而不是模型预测的 token。避免错误累计
    '''
    # 初始化token的总数和损失
    total_tokens = 0.
    total_loss = 0.

    # 遍历整个数据集（数据为batch的形式）
    for batch in tqdm(data):
        # 模型前向传播，得到预测结果out
        decoder_out = model(batch.src, batch.decoder_input, batch.src_mask, batch.decoder_input_mask)

        # 使用loss_compute计算损失
        loss = loss_compute(decoder_out, batch.target_label, batch.num_tokens)

        # 累加损失和有效tokens的数量
        total_loss += loss
        total_tokens += batch.num_tokens

    # 计算每个token的平均损失
    mean_token_loss = total_loss / total_tokens

    return mean_token_loss

def train(train_data, val_data, model, model_par, criterion, optimizer):
    '''
    训练并保存模型
    train_data: 训练数据集
    val_data: 验证数据集
    model: 模型
    model_par: 模型并行化
    criterion: 损失计算函数
    optimizer: 优化器
    '''
    # best_bleu_score初始化
    best_bleu_score = -float('inf')  # 初始最佳BLEU分数为负无穷
    # 创建保存权重的路径
    exp_folder, weights_folder = create_exp_folder()

    # 开始训练循环，迭代每个epoch
    for epoch in range(0, config.epoch_num):
        logging.info(f"第{epoch}轮模型训练与验证")

        # 设置模型为训练模式
        model.train()
        
        # 进行一个epoch的训练，返回当前的训练损失
        # model 中截止到 decoder 的输出，model.generator 单独传入
        loss_train_compute = ChunkedGeneratorLossCompute(model.generator, criterion, config.device_id, optimizer)
        train_loss = run_epoch(train_data, model_par, loss_train_compute)

        # 设置模型为评估模式（即不计算梯度，优化）
        model.eval()
        # 进行一个epoch的验证，返回当前的验证损失
        loss_eval_compute = ChunkedGeneratorLossCompute(model.generator, criterion, config.device_id, None)
        val_loss = run_epoch(val_data, model_par, loss_eval_compute)

        # 计算模型在验证集（val_data）上的BLEU分数
        bleu_score = evaluate(val_data, model)
        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, Bleu Score: {bleu_score:.2f}\n")

        # 如果当前epoch的模型的BLEU分数更优，则保存最佳模型
        if bleu_score > best_bleu_score:
            # 如果之前已存在最优模型，先删除
            if best_bleu_score != -float('inf'):
                old_model_path = f"{weights_folder}/best_bleu_{best_bleu_score:.2f}.pth"
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            model_path_best = f"{weights_folder}/best_bleu_{bleu_score:.2f}.pth"
            # 保存当前模型的状态字典到指定路径
            torch.save(model.state_dict(), model_path_best)
            # 更新最佳BLEU分数
            best_bleu_score = bleu_score
        
        # 保存当前模型（最后一次训练）
        if epoch == config.epoch_num:  # 判断是否达到设定的训练轮数
            model_path_last = f"{weights_folder}/last_bleu_{bleu_score:.2f}.pth"  # 构建模型保存路径，包含BLEU分数
            torch.save(model.state_dict(), model_path_last)  # 保存模型的状态字典
    
    return model_path_best

def evaluate(data, model):
    '''
    train 过程中，在 val_data 上用训练好的模型进行预测，返回 BLEU 分数，用于判断模型是否收敛和保存最佳模型
    真正的自回归生成，存在错误累计问题，无法保证翻译质量
    '''
    sp_chn = chinese_tokenizer_load()  # 加载中文分词器
    trg = []  # 存储目标句子（真实句子）
    res = []  # 存储模型翻译的结果

    # 禁用梯度计算，节省内存和计算
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):  # 使用tqdm显示进度条
            # 翻译目标使用完整的文本
            cn_sent = batch.tgt_text  # 获取当前批次的中文句子
            # 被翻译的源语言句子使用的是 token ID 序列，因为要进行前向推理
            # 添加 BOS 和 EOS 标记，并进行 padding
            src = batch.src   # 获取当前批次的源语言（英文）句子
            src_mask = (src != 0).unsqueeze(-2)    # 为源语言句子创建mask，排除padding部分

            # 使用束搜索生成模型翻译结果
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
            
            # `decode_result`是一个包含多个翻译结果的列表，取最优结果
            decode_result = [h[0] for h in decode_result]
            # 解码后的id转为中文句子
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)  # 将当前批次的真实句子添加到`trg`中
            res.extend(translation)  # 将模型的翻译结果添加到`res`中

    # 计算BLEU分数，使用SacreBLEU工具库
    trg = [trg]  # 真实目标句子
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')  # 计算BLEU分数
    return float(bleu.score)  # 返回BLEU分数

def test(data, model, criterion, model_path_best):
    '''
    test 过程中，在 test_data 上用训练好的模型进行预测，返回 BLEU 分数
    验证训练后模型的效果
    '''
    with torch.no_grad():
        # 从训练完成的 model_path 中加载最佳模型
        model.load_state_dict(torch.load(model_path_best))
        # 设置并行化
        model_par = torch.nn.DataParallel(model)
        # 设置模型为评估模式
        model.eval()

        # 开始预测
        loss_test_compute = ChunkedGeneratorLossCompute(model.generator, criterion, config.device_id, None)
        # 进行一个epoch的测试，返回当前的测试损失
        test_loss = run_epoch(data, model_par, loss_test_compute)
        # 计算模型在测试集（test_data）上的BLEU分数
        bleu_score = evaluate(data, model)
        logging.info(f"Test loss: {test_loss:.3f}, Bleu Score: {bleu_score:.2f}\n")


def run():
    '''
    主运行函数，用于整体流程搭建
    '''
    # 创建训练数据集、评估、验证数据集
    train_dataset = MachineTranslationDataset(config.train_data_path)
    val_dataset = MachineTranslationDataset(config.val_data_path)
    test_dataset = MachineTranslationDataset(config.test_data_path)

    # 创建数据加载器，使用标准方法 DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.num_layers,
                       config.d_token_embedding, config.d_k, config.d_ffn, config.num_heads, config.dropout)
    
    # 将模型包装成数据并行模式,这样可以在多个GPU上并行处理数据，提高训练效率
    model_par = torch.nn.DataParallel(model)

    # 选择损失函数
    # CrossEntropyLoss是常见的分类问题损失函数，ignore_index=0表示忽略填充部分
    # reduction='sum'表示计算损失时会对所有token的损失求和
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # 调用get_std_opt函数获取标准的Noam优化器
    optimizer = get_std_opt(model)

    # 开始训练
    test_model_path = train(train_dataloader, val_dataloader, model, model_par, criterion, optimizer)
    # 测试模型
    # # 可以进行单独测试
    # test_model_path = config.test_model_path
    test(test_dataloader, model, criterion, test_model_path)


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')
    run()