import yaml
import torch
from typing import Dict, Any

class Config:
    """配置类，用于加载和管理YAML配置文件"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置类
        :param config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config = self._load_config()
        self._setup_device()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def _setup_device(self):
        """设置设备配置"""
        gpu_id = self._config['device']['gpu_id']
        if gpu_id != '':
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device('cpu')
    
    def get(self, key_path: str, default=None):
        """
        获取配置值，支持嵌套键访问
        :param key_path: 配置键路径，如 'model.d_token_embedding'
        :param default: 默认值
        :return: 配置值
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"配置键 '{key_path}' 不存在")
    
    '''
    使用 @property 装饰器，将方法转换为属性，方便使用
    可以像访问属性一样访问方法，不用携带括号
    eg: config.d_token_embedding
    '''
    
    # 模型超参数属性
    @property
    def d_token_embedding(self):
        return self.get('model.d_token_embedding')
    
    @property
    def num_heads(self):
        return self.get('model.num_heads')
    
    @property
    def num_layers(self):
        return self.get('model.num_layers')
    
    @property
    def d_k(self):
        return self.get('model.d_k')
    
    @property
    def d_ffn(self):
        return self.get('model.d_ffn')
    
    @property
    def dropout(self):
        return self.get('model.dropout')
    
    # 词汇表配置属性
    @property
    def src_vocab_size(self):
        return self.get('vocab.src_vocab_size')
    
    @property
    def tgt_vocab_size(self):
        return self.get('vocab.tgt_vocab_size')
    
    @property
    def padding_idx(self):
        return self.get('vocab.padding_idx')
    
    @property
    def bos_idx(self):
        return self.get('vocab.bos_idx')
    
    @property
    def eos_idx(self):
        return self.get('vocab.eos_idx')
    
    # 训练配置属性
    @property
    def batch_size(self):
        return self.get('training.batch_size')
    
    @property
    def epoch_num(self):
        return self.get('training.epoch_num')
    
    @property
    def lr_factor(self):
        return self.get('training.lr_factor')

    
    @property
    def warmup_steps(self):
        return self.get('training.warmup_steps')
    
    @property
    def tokenizer_path(self):
        return self.get('training.tokenizer_path')
    
    # 解码配置属性
    @property
    def max_len(self):
        return self.get('decoding.max_len')
    
    @property
    def beam_size(self):
        return self.get('decoding.beam_size')
    
    # 路径配置属性
    @property
    def dataset_path(self):
        return self.get('paths.dataset_path')
    
    @property
    def train_data_path(self):
        return self.get('paths.train_data_path')
    
    @property
    def val_data_path(self):
        return self.get('paths.val_data_path')
    
    @property
    def test_data_path(self):
        return self.get('paths.test_data_path')
    
    @property
    def model_path(self):
        return self.get('paths.model_path')
    
    @property
    def test_model_path(self):
        return self.get('paths.test_model_path')
    
    # 设备配置属性
    @property
    def gpu_id(self):
        return self.get('device.gpu_id')
    
    @property
    def device_id(self):
        return self.get('device.device_id')

# 全局配置实例
config_path = './config/config.yaml'
config = Config(config_path)