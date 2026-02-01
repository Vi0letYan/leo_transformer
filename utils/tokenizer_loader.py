import sentencepiece as spm
from utils.config import config


def chinese_tokenizer_load():
    """
    加载中文分词器模型
    该函数用于加载预训练的中文SentencePiece分词器模型，用于文本预处理和分词。
    返回:
        sp_zh: 加载好的SentencePieceProcessor对象，可用于中文文本的分词处理
    使用方法:
        tokenizer = chinese_tokenizer_load()
        tokens = tokenizer.tokenize("中文文本")
    """
    # 创建SentencePieceProcessor对象
    sp_zh = spm.SentencePieceProcessor()
    # 加载预训练的中文分词模型
    zh_model_path = config.tokenizer_path + '/zh.model'
    sp_zh.Load(zh_model_path)
    # 返回加载好的分词器对象
    return sp_zh

def english_tokenizer_load():
    """
    加载英文分词器模型
    该函数用于加载英文的SentencePiece分词器模型，该模型用于将英文文本转换为token序列。
    返回:
        sp_eng: 加载好的SentencePieceProcessor对象，可用于英文文本的分词处理
    使用方法:
        tokenizer = english_tokenizer_load()
        tokens = tokenizer.tokenize("英文文本")
    """
    # 创建SentencePieceProcessor对象
    sp_eng = spm.SentencePieceProcessor()
    # 加载预训练的英文分词模型
    eng_model_path = config.tokenizer_path + '/eng.model'
    sp_eng.Load(eng_model_path)
    # 返回加载好的分词器对象
    return sp_eng