import torch
from utils.data_loader import subsequent_mask

import torch.nn.functional as F


class Beam:
    """ Beam search 单个束的状态管理 """

    def __init__(self, size, pad, bos, eos, device=False):
        """
        初始化Beam对象
        size: beam宽度（保留多少个候选）
        pad/bos/eos: 特殊token的ID
        device: 计算设备
        """
        self.size = size
        self._done = False  # 是否已完成（遇到EOS）
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        # 每个beam的累积log概率分数，初始为0
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []  # 保存每步的历史分数

        # 反向指针列表，用于回溯重建序列
        self.prev_ks = []

        # 每个时刻选择的词ID列表
        # 初始化为 [BOS, PAD, PAD, ..., PAD]，只有第一个beam从BOS开始
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        """获取当前所有beam的解码序列"""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """获取最近一步的反向指针（每个beam来自哪个父beam）"""
        return self.prev_ks[-1]

    @property
    def done(self):
        """返回是否已完成"""
        return self._done

    def advance(self, word_logprob):
        """
        执行一步beam扩展
        word_logprob: (beam_size, vocab_size) 当前每个beam对每个词的log概率
        返回: 是否完成
        """
        num_words = word_logprob.size(1)  # 词表大小

        # 计算扩展后的累积分数
        if len(self.prev_ks) > 0:
            # 非首步：历史分数 + 当前词概率
            # scores: (beam_size,) -> (beam_size, 1) -> (beam_size, vocab_size)
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # 首步：只用第一个beam（其他beam是PAD，无意义）
            beam_lk = word_logprob[0]

        # 展平为一维，从 beam_size*vocab_size 个候选中选top-k
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        # 保存历史分数，更新当前分数
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # 从展平索引还原：来自哪个父beam，选了哪个词
        # best_scores_id 范围 [0, beam_size*vocab_size)
        prev_k = torch.div(best_scores_id, num_words, rounding_mode='floor')  # 父beam索引
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)  # 词ID = 余数

        # 终止条件：最优beam（索引0）的最后一个词是EOS
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """按分数降序排序，返回(排序后分数, 对应索引)"""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """获取第二好的分数和索引（用于某些场景）"""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """获取当前所有beam的完整解码序列"""
        if len(self.next_ys) == 1:
            # 首步，只有BOS
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            # 按分数排序，回溯构建每个beam的序列
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]  # 加上BOS前缀
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """
        回溯构建第k个beam的假设序列
        k: beam索引
        返回: 词ID列表（不含BOS）
        """
        hyp = []
        # 从最后一步向前回溯
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])  # 当前步选的词
            k = self.prev_ks[j][k]  # 跳转到父beam

        # 反转得到正序，转为python int列表
        return list(map(lambda x: x.item(), hyp[::-1]))


def beam_search(model, src, src_mask, max_len, pad, bos, eos, beam_size, device):
    """
    批量beam search主函数
    model: Transformer模型
    src: 源序列 (batch_size, src_len)
    src_mask: 源序列mask
    max_len: 最大解码长度
    pad/bos/eos: 特殊token ID
    beam_size: beam宽度
    device: 计算设备
    """

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """
        建立实例索引到张量位置的映射
        例如活跃实例[0,2,5] -> {0:0, 2:1, 5:2}
        """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """
        从张量中提取仍活跃的实例部分
        beamed_tensor: (n_prev_active_inst * beam_size, ...)
        返回: (n_curr_active_inst * beam_size, ...)
        """
        _, *d_hs = beamed_tensor.size()  # 获取除第一维外的形状
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        # 先reshape为(实例数, beam_size*其他维度)，按实例选取，再reshape回来
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        """
        收集仍活跃实例的编码和mask
        过滤掉已完成的实例，减少计算量
        """
        n_prev_active_inst = len(inst_idx_to_position_map)
        # 将实例索引转换为张量位置
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        # 提取活跃部分
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)

        return active_src_enc, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        """
        执行一步解码
        返回: 未完成的实例索引列表
        """

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            """准备当前解码序列，用于decoder输入"""
            # 收集所有未完成beam的当前序列
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            # (batch_size, beam_size, seq_len)
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            # 展平为 (batch_size * beam_size, seq_len)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
            """调用decoder预测下一个词的概率分布"""
            # 确保维度匹配
            assert enc_output.shape[0] == dec_seq.shape[0] == src_mask.shape[0]
            # 解码：enc_output是编码器输出，dec_seq是当前解码序列
            out = model.decode(enc_output, src_mask,
                               dec_seq,
                               subsequent_mask(dec_seq.size(1))  # 因果mask
                               .type_as(src.data))
            # 只取最后一个位置的输出，通过generator得到词概率，并进行 log_softmax 操作
            # generator 为了适配 crossentropyloss 的输入，所以输出没有进行 log_softmax 操作
            word_logprob = model.generator(out[:, -1])
            word_logprob = F.log_softmax(word_logprob, dim=-1)
            # reshape为 (实例数, beam_size, vocab_size)
            word_logprob = word_logprob.view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            """更新每个beam状态，返回未完成的实例列表"""
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                # 用当前词概率更新beam状态
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    # 未完成的实例继续解码
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        # 步骤1: 准备decoder输入序列
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

        # 步骤2: 预测下一词概率
        word_logprob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)

        # 步骤3: 更新beam状态，收集未完成实例
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        """收集所有实例的最终假设和分数"""
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            # 按分数排序
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]  # 取前n_best个分数

            # 回溯构建前n_best个假设
            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    # ========== 主流程开始 ==========
    with torch.no_grad():
        # 步骤1: 编码源序列
        src_enc = model.encode(src, src_mask)

        # 步骤2: 为beam search复制编码结果
        # 每个实例复制beam_size份
        NBEST = beam_size
        batch_size, sent_len, h_dim = src_enc.size()
        # (batch, seq, dim) -> (batch*beam, seq, dim)
        src_enc = src_enc.repeat(1, beam_size, 1).view(batch_size * beam_size, sent_len, h_dim)
        # mask同样复制
        src_mask = src_mask.repeat(1, beam_size, 1).view(batch_size * beam_size, 1, src_mask.shape[-1])

        # 步骤3: 为每个batch实例创建Beam对象
        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]

        # 步骤4: 初始化活跃实例追踪
        active_inst_idx_list = list(range(batch_size))  # 所有实例都活跃
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # 步骤5: 逐步解码
        for len_dec_seq in range(1, max_len + 1):
            # 执行一步解码，获取仍活跃的实例
            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)

            # 所有实例都完成了，提前退出
            if not active_inst_idx_list:
                break

            # 过滤掉已完成的实例，减少后续计算
            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

    # 步骤6: 收集最终结果
    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, NBEST)

    return batch_hyp, batch_scores