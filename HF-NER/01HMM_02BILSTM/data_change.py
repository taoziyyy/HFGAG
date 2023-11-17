import json
import os


# 实现对json的数据读取并转换成BIO格式
# 目前用不到
def read_change(input_path, out_path, lowercase=True):  # 默认所有单词均统一为小写模式
    with open(input_path, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
    for data in data_all:
        context = data['context']  # 此时读取出来的json对象为字典类型
        sentence_word = context.split(' ')  # 通过空格进行分词
        if lowercase:
            sentence_word = [word.lower() for word in sentence_word]
        labels = data['labels']
        sentence_bio_label = ['O' for word in sentence_word]  # 先用O覆盖全句
        for label in labels:  # 依次处理每一种label
            entity_label = label['entity_label']
            span_list = label['span_list']
            for span in span_list:
                start = span[0][0]
                end = span[0][1]  # 找到一个entity的
                sentence_bio_label[start] = 'B-' + entity_label
                if end > start:
                    for i in range(start+1, end+1):  # 这里右开区间是个坑
                        sentence_bio_label[i] = 'I-' + entity_label
        data_all_bio = list(zip(sentence_word, sentence_bio_label))
        with open(out_path, 'a', encoding='utf-8') as f:
            for data_bio in data_all_bio:
                f.write('{} {}'.format(data_bio[0], data_bio[1]))
                f.write('\n')
            f.write('\n')  # 每一句的结束需要换行


def main():
    folder = os.path.exists('data/conll03_BIO')  # 创建一个新的文件夹目录放置转换为BIO格式的文件
    if not folder:
        os.makedirs('data/conll03_BIO')
    read_change('data/conll03/train.json', 'data/conll03_BIO/train.txt')
    read_change('data/conll03/test.json', 'data/conll03_BIO/test.txt')
    read_change('data/conll03/dev.json', 'data/conll03_BIO/dev.txt')


if __name__ == '__main__':
    main()