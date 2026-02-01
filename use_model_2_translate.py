import torch
import numpy as np

from model_codes.transformer_model import make_model
from utils.config import config
from utils.tokenizer_loader import english_tokenizer_load, chinese_tokenizer_load
from utils.beam_search import beam_search

def translate():

    model = make_model(
            config.src_vocab_size,
            config.tgt_vocab_size,
            config.num_layers,
            config.d_token_embedding,
            config.d_k,
            config.d_ffn,
            config.num_heads,
            config.dropout
            )
    # 获取 BOS 和 EOS 的索引
    BOS = english_tokenizer_load().bos_id()
    EOS = english_tokenizer_load().eos_id()
    
    while True:
        sent = input("请输入英文句子进行翻译：")

        # 将输入的句子转化为 token IDs，添加 BOS 和 EOS 标记
        src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]

        # 将句子转换为长整型Tensor，并发送到指定的设备（如GPU或CPU）
        model_input = torch.LongTensor(np.array(src_tokens)).to(config.device)

        with torch.no_grad():
            model.load_state_dict(torch.load(config.model_path))
            model.eval()
        
            # 创建源句子的掩码（mask），以确保填充的部分不会参与计算
            src_mask = (model_input != 0).unsqueeze(-2)

            # 使用束搜索（beam search）进行解码
            decode_result, _ = beam_search(
                model,
                model_input,
                src_mask,
                config.max_len,
                config.padding_idx,
                config.bos_idx,
                config.eos_idx,
                config.beam_size,
                config.device
            )

            # 从解码结果中提取最优结果
            decode_result = [h[0] for h in decode_result]

            # 使用中文分词器将解码结果的id转化为实际的中文词语
            translation = [chinese_tokenizer_load().DecodePieces(_s) for _s in decode_result]

            print("翻译结果：", translation[0])

if __name__ == "__main__":
    translate()