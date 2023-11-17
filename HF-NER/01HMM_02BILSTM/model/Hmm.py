import torch


class HMM(object):
    def __init__(self, N, M):
        """Args:
            N: 状态数，这里对应存在的标注的种类
            M: 观测数，这里对应有多少不同的字
        """
        self.N = N
        self.M = M
        self.A = torch.zeros(N, N)  # 状态转移概率矩阵
        self.B = torch.zeros(N, M)  # 观测概率矩阵
        self.Pi = torch.zeros(N)  # 初始状态概率

    def train(self, words_list, tags_list, word2id, tag2id):

        assert len(tags_list) == len(words_list)

        # 估计转移概率矩阵
        for tags in tags_list:
            seq_len = len(tags)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tags[i]]
                next_tagid = tag2id[tags[i+1]]
                self.A[current_tagid][next_tagid] += 1

        self.A[self.A == 0.] = 1e-10  # 防止某个元素为0
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 估计观测概率矩阵
        for tags, words in zip(tags_list, words_list):
            assert len(tags) == len(words)
            for tag, word in zip(tags, words):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1  # 记录出隐状态tag_id对应的word_id
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 估计初始状态概率
        for tags in tags_list:
            init_tagid = tag2id[tags[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def test(self, word_lists, word2id, tag2id):
        pred_tags_list = []
        for word_list in word_lists:
            pred_tags = self.decoding(word_list, word2id, tag2id)
            pred_tags_list.append(pred_tags)
        return pred_tags_list

    def decoding(self, words, word2id, tag2id):
        # 问题:整条链很长的情况下，十分多的小概率相乘，最后可能造成下溢，因此采用对数操作使得小概率变成负数
        #  同时相乘操作也变成简单的相加操作
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        seq_len = len(words)
        viterbi = torch.zeros(self.N, seq_len)

        backpointer = torch.zeros(self.N, seq_len).long()

        start_wordid = word2id.get(words[0], None)  # 这里使用get还不是直接取值是为了防止不存在时报错
        B_t = B.t()
        if start_wordid is None:
            # 如果字不在字典里，则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.N) / self.N)
        else:
            bt = B_t[start_wordid]  # 找到每个隐状态对应的概率
        viterbi[:, 0] = Pi + bt  # 这里采用加法的原因是进行了对数操作
        backpointer[:, 0] = -1

        for step in range(1, seq_len):
            word_id = word2id.get(words[step], None)
            # 处理字不在字典中的情况
            # bt是在t时刻字为wordid时，状态的概率分布
            if word_id is None:
                # 如果字不再字典里，则假设状态的概率分布是均匀的
                bt = torch.log(torch.ones(self.N) / self.N)
            else:
                bt = B_t[word_id]  # 否则从观测概率矩阵中取bt
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(viterbi[:, step-1] + A[:, tag_id], dim=0)
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        # 终止， t=seq_len 即 viterbi[:, seq_len]中的最大概率，就是最优路径的概率
        best_path_prob, best_path_pointer = torch.max(viterbi[:, seq_len-1], dim=0)

        # 回溯，求最优路径
        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        # 将tag_id组成的序列转化为tag
        assert len(best_path) == len(words)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tags = [id2tag[id_] for id_ in reversed(best_path)]

        return tags