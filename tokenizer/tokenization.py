# 导入 sentencepiece 库：用于 无监督训练子词（BPE/Unigram）模型、后续编码/解码
import sentencepiece as spm

def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    重要说明（官方参数文档可查）：
    https://github.com/google/sentencepiece/blob/master/doc/options.md

    参数含义：
    - input_file: 原始语料文件路径（每行一句，SentencePiece 会做 Unicode NFKC 规范化（统一字符形态，如全角转半角、异体字归一））
                  支持多文件逗号拼接：'a.txt,b.txt'
    - vocab_size: 词表大小，如 8000 / 16000 / 32000，不能大于数据集中的 “唯一词（原始未分词状态）数量”
    - model_name: 模型前缀名，最终会生成 <model_name>.model 和 <model_name>.vocab
                  <model_name>.model：子词列表、拆分规则、符号 ID 映射，就是后续会用到的 tokenizer 模型文件
                  既是name，也是路径，只要在name前加上路径即可
    - model_type: 模型类型：unigram（默认）/ bpe / char / word
                  注意：若使用 word，需要你在外部先分好词（预分词）
                  在 unigram （效果最好） 模型下，中文和英文的最小分词单位本质都是 “字符”（中文是单个汉字，英文是单个字母），再基于概率合并高频汉字组合
    - character_coverage: 覆盖的字符比例
        * 中文/日文等字符集丰富语言建议 0.9995：0.9995的子词为核心分词，其余为稀有子词
        * 英文等字符集小的语言建议 1.0
    """
    # 这里使用“字符串命令”式的调用来指定训练参数
    # 固定 4 个特殊符号的 id：<pad>=0, <unk>=1, <bos>=2, <eos>=3
    # 这与下游 Transformer 常用配置一致，便于对齐
    input_argument = (
        '--input=%s '
        '--vocab_size=%s '
        '--model_prefix=%s '
        '--model_type=%s '
        '--character_coverage=%s '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    )

    # 将传入参数填充到命令字符串中
    cmd = input_argument % (input_file, vocab_size, model_name, model_type, character_coverage)

    # 开始训练；会在当前工作目录下生成 <model_name>.model / <model_name>.vocab
    # 通过 “训练”，让模型从你的语料中 “学” 出适合这份语料的拆分方式 —— 这也是为什么不同语料训练出的 SentencePiece 模型，分词结果会不一样
    # 说是训练，实际上是一种算法，根据我们提供的数据集，产生相对应的分词结果
    # 因为不同的数据集会产生不同的分词结果，看起来如同训练
    spm.SentencePieceTrainer.Train(cmd)

def run(tokenizer_path):
    # 英文 tokenizer config
    # 英文语料：一行一句
    en_input = './translation_en_zh/corpus_en.txt'
    # 次表大小：不能大于数据集中的 “唯一词（原始未分词状态）数量”。常见为 16k/32k
    en_vocab_size = 32000
    # 输出前缀：.model .vocab 文件的名称及路径
    # .model 是最后会使用的 tokenizer， .vocab 是供人工查阅的文件
    # .model 包含：子词列表、拆分规则、符号 ID 映射
    en_model_name = tokenizer_path + '/' + 'eng'
    # 分词类型：unigram（效果最好但最慢：动态选最优：对新文本，枚举所有可能的子词组合，选 “概率乘积最大” 的组合）
    en_model_type = 'unigram'
    # 英文字符集较小，所有子词都为核心子词
    en_character_coverage = 1.0

    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    # 中文 tokenizer config
    zh_input = './translation_en_zh/corpus_zh.txt'
    zh_vocab_size = 32000
    zh_model_name = tokenizer_path + '/' + 'zh'
    zh_model_type = 'unigram'
    # 中文字符集较大，设定 0.9995 的子词为核心子词，其余低概率冷僻字映射为 <unk>
    zh_character_coverage = 0.9995

    train(zh_input, zh_vocab_size, zh_model_name, zh_model_type, zh_character_coverage)

def test(tokenizer_path):
    # 加载并调用已经训练好的 spm 模型进行编码/解码
    sp = spm.SentencePieceProcessor()
    text = '日本首相高市早苗破坏中日关系'

    # 加载中文tokenizer模型
    zh_model_path = tokenizer_path + '/' + 'zh.model'
    sp.Load(zh_model_path)

    # 编码子词片段，返回一个列表，列表中每个元素是一个子词
    print(sp.EncodeAsPieces(text))

    # 编码为 id，返回一个列表，列表中每个元素是一个id
    print(sp.EncodeAsIds(text))

    # 给一串 id， 解码回文本，返回一个字符串
    a = [12907, 277, 7419, 7318, 18384, 28724]
    print(sp.DecodeIds(a))

if __name__ == "__main__":
    tokenizer_path = './tokenizer'
    # run(tokenizer_path)
    test(tokenizer_path)

