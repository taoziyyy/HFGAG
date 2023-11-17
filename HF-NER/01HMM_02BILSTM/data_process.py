# 构建单词和标签语料库
def build_corpus(split, make_vocab=True):
    # data_path = 'data/conll03_BIO/' + split + '.txt'
    data_path = 'data/2023-4-data/' + split + '.txt'
    # 读取数据
    words_list = []
    tags_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        for line in f:
            if line != '\n':  # 列表内嵌列表实现多个句子在一个列表中
                v = line.strip('\n').split()  # 把换行符去掉
                words.append(v[0])
                # tags.append(v[3])
                tags.append(v[1])
            else:
                words_list.append(words)
                tags_list.append(tags)
                words = []  # 清空列表
                tags = []

    if make_vocab:  # 在训练集上建立映射
        word2id = build_map(words_list, False)
        tag2id = build_map(tags_list, True)
        return words_list, tags_list, word2id, tag2id
    else:
        return words_list, tags_list

# 构建单词的映射库
def build_map(lists, tag):
    maps = {}
    if tag:  # 标记是否是标签，因为0映射的数据不同，此处提前规定好0标签的对应值方便之后在lstm补全时用
        maps['O'] = 0  # 把O放在第一位上
    else:
        maps['<pad>'] = 0  # 把pad放在第一位上
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps

# 构建单词转映射的函数
def word_tag_id(word_list, tag_list, word_map, tag_map):
    # 把str型转换成id
    words_data = []
    tags_data = []
    for words, tags in zip(word_list, tag_list):
        word_data = []
        tag_data = []
        for word, tag in zip(words, tags):
            try:
                word_data.append(word_map[word])
            except:
                word_data.append(word_map['<unk>'])
            tag_data.append(tag_map[tag])
        words_data.append(word_data)
        tags_data.append(tag_data)
    return words_data, tags_data


