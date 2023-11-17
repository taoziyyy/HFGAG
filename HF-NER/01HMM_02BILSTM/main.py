from model.Bilstm import BiLSTM
from data_process import build_corpus, word_tag_id
import torch
import torch.nn as nn
import torch.utils.data as tud
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from evaluate import metric, output_result
from model.Hmm import HMM


torch.manual_seed(1024)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class my_dataset(tud.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

# 需要统一序列的长度。使用每批中最大的序列长度作为固定长度，为短序列进行填充
def collate_fn(batch):
    words_len = [len(sequence[0]) for sequence in batch]  # 0放的文本，1放的标签
    max_len = max(words_len)
    words_list, tags_list = [], []
    for words, tags in batch:
        if len(words) < max_len:
            words.extend([0] * (max_len - len(words)))
            tags.extend([0] * (max_len - len(tags)))
        words_list.append(words)
        tags_list.append(tags)
    return torch.LongTensor(words_list), torch.LongTensor(tags_list), torch.LongTensor(words_len)

# 加载数据
def build_loader(data, batch_size, is_shufflue):
    dataset = my_dataset(data)
    my_dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=is_shufflue, collate_fn=collate_fn)
    return my_dataloader

# 将单词映射为id,将<unk>的单词设置固定映射值
def extend_maps(word2id):
    word2id['<unk>'] = len(word2id)
    return word2id

# 将标签映射值转换回标签真实值
def change_map(tag2id):
    id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
    return id2tag

# 训练hmm模型
def hmm_train(train_data, dev_data, test_data, word2id, tag2id):
    hmm = HMM(len(tag2id), len(word2id))
    train_words_list, train_tags_list = train_data
    dev_words_list, dev_tags_list = dev_data
    test_words_list, test_tags_list = test_data
    hmm.train(train_words_list, train_tags_list, word2id, tag2id)
    id2tag = change_map(tag2id)

    dev_preds_list = hmm.test(dev_words_list, word2id, tag2id)
    info, class_info = metric(dev_preds_list, dev_tags_list, id2tag)
    print('验证集上结果分析:')
    print("precision = {:.4f}, recall = {:.4f}, f1_score = {:.4f}".format(info['acc'], info['recall'], info['f1']))

    test_preds_list = hmm.test(test_words_list, word2id, tag2id)
    info, class_info = metric(test_preds_list, test_tags_list, id2tag)
    output_result((test_words_list, test_preds_list), test_tags_list, is_HMM=True, id2tag=id2tag)
    print('测试集上结果分析:')
    print("precision = {:.4f}, recall = {:.4f}, f1_score = {:.4f}".format(info['acc'], info['recall'], info['f1']))


def evaluate(model, dataLoader, tag2id):
    with torch.no_grad():
        preds_list = []
        true_tags_list = []
        for batch in dataLoader:
            words, tags, lengths = batch
            words = words.to(device)
            tags = tags.to(device)
            scores = model(words, lengths)
            preds = torch.argmax(scores, dim=2)
            preds_list.extend(preds.data.cpu().tolist())
            true_tags_list.extend(tags.data.cpu().tolist())
    id2label = change_map(tag2id)
    info, class_info = metric(preds_list, true_tags_list, id2label)
    print("precision = {:.4f}, recall = {:.4f}, f1_score = {:.4f}".format(info['acc'], info['recall'], info['f1']))
    return preds_list


# 训练lstm模型
def lstm_train(train_data, dev_data, test_data, word2id, tag2id, lr, epochs):
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bi_lstm_model = BiLSTM(vocab_size, emb_size=128, hidden_size=128, out_size=out_size).to(device)
    criterion = nn.CrossEntropyLoss()
    params = bi_lstm_model.parameters()
    optimizer = torch.optim.Adam(params=params, lr=lr)
    last_loss = 100
    torch.backends.cudnn.enabled=False
    for epoch in range(epochs):
        epoch_loss = []  # 求平均loss
        for train_batch in train_data:
            optimizer.zero_grad()
            words, tags, lengths = train_batch
            words = words.to(device)
            tags = tags.to(device)
            scores = bi_lstm_model(words, lengths)
            scores = pack_padded_sequence(scores, lengths, batch_first=True, enforce_sorted=False).data
            tags = pack_padded_sequence(tags, lengths, batch_first=True, enforce_sorted=False).data
            loss = criterion(scores, tags)
            loss.backward()
            optimizer.step()
            epoch_loss.append((loss.item()))
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print('epoch {} loss: {}'.format(epoch, epoch_loss))
        bi_lstm_model.eval()
        if epoch > 2 and abs(last_loss - epoch_loss) < 1e-3:
            break
        last_loss = epoch_loss
    print('验证集上结果分析:')
    evaluate(bi_lstm_model, dev_data, tag2id)
    print('测试集上结果分析:')
    test_preds_list = evaluate(bi_lstm_model, test_data, tag2id)
    with open('epoch{}.pth'.format(epochs), 'wb') as f:
        torch.save(bi_lstm_model.state_dict(), f)
    return test_preds_list


def main():
    # 构建训练集上单词和标签的语料库
    train_words_list, train_tags_list, word2id, tag2id = build_corpus("train")  # 载入数据
    # 获得验证集和测试集上单词和标签列表
    dev_words_list, dev_tags_list = build_corpus("dev", make_vocab=False)
    # test_words_list, test_tags_list = build_corpus("test", make_vocab=False)
    test_words_list, test_tags_list = build_corpus("train", make_vocab=False)

    # 将单词映射为id,将<unk>的单词设置固定映射值
    word2id = extend_maps(word2id)

    print('--------------------------------------------------Hmm-------------------------------------------------------')
    hmm_train((train_words_list, train_tags_list), (dev_words_list, dev_tags_list), (test_words_list, test_tags_list),
              word2id, tag2id)
    print('------------------------------------------------BiLSTM------------------------------------------------------')
    train_data, dev_data, test_data = [], [], []
    train_word_data, train_tag_data = word_tag_id(train_words_list, train_tags_list, word2id, tag2id) # 构建单词转映射的函数
    dev_word_data, dev_tag_data = word_tag_id(dev_words_list, dev_tags_list, word2id, tag2id)
    test_word_data, test_tag_data = word_tag_id(test_words_list, test_tags_list, word2id, tag2id)
    for word, tag in zip(train_word_data, train_tag_data): # 在for循环里zip()函数用来并行遍历列表。
        train_data.append([word, tag])
    for word, tag in zip(dev_word_data, dev_tag_data):
        dev_data.append([word, tag])
    for word, tag in zip(test_word_data, test_tag_data):
        test_data.append([word, tag])
    train_dataLoader = build_loader(train_data, batch_size=16, is_shufflue=True)
    dev_dataLoader = build_loader(dev_data, batch_size=16, is_shufflue=False)
    test_dataLoader = build_loader(test_data, batch_size=16, is_shufflue=False)
    test_preds_list = lstm_train(train_dataLoader, dev_dataLoader, test_dataLoader, word2id, tag2id, lr=0.01, epochs=5)
    id2tag = change_map(tag2id)# 将标签映射值转换回标签真实值
    output_result((test_words_list, test_preds_list), test_tags_list, is_HMM=False, id2tag=id2tag)
import torch

if __name__ == '__main__':
    main()
