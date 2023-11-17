from utils.tokenizer import tokenize
from utils.ann_utils import *
from utils.sequence_utils import *
import numpy as np
import pandas as pd
import sys
import os

def get_token_offset(tokens, sentences):
    token_offsets = []
    for i, sentence in enumerate(sentences):
        sentence_token_offsets = []
        sentence_tokens = tokens[i]
        for j in range(len(sentence_tokens)):
            if j == 0:
                token_offset = sentence.find(sentence_tokens[j])
            else:
                offset_begin = last_token_offset + len(sentence_tokens[j-1])
                sentence_to_find = sentence[offset_begin:]
                additional_offset = sentence_to_find.find(sentence_tokens[j])
                token_offset = offset_begin + additional_offset
            sentence_token_offsets.append(token_offset)
            last_token_offset = token_offset
        token_offsets.append(sentence_token_offsets)
    return token_offsets

def get_sent_offset(sentences, text):
    sent_offset = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            offset = 0
        else:
            offset = last_offset + len(last_sentence) + 1
        if sentence != text[offset: offset+len(sentence)]:
            offset = offset + text[offset:].find(sentence)
        sent_offset.append(offset)        
        last_offset = offset
        last_sentence = sentence
    return sent_offset

def load_documents(text_path, sentence_path, ner_path, word_index):
    
    documents = []
    gdas_total = []
    seqs = []
    features = []
    text_file = open(text_path, "r")
    sentence_file = open(sentence_path, "r")
    ner_file = open(ner_path, "r")
#     no_packs = 0
    pmid = ""
    while (1):
        try:
            line = text_file.readline()
            if line == '':
                break
            pmid = line.strip()
            title = text_file.readline().strip()
            abstract = text_file.readline().strip()
            text = title + " " + abstract
            text_file.readline()

            sentences = []
            tokens =[]
            line = sentence_file.readline()
            if line == "\n":
                sentence_file.readline()
            while (1):
                line = sentence_file.readline()
                if line == "\n":
                    break
                sentence = line.strip()
                sentences.append(sentence)
                sentence_tokens = tokenize(sentence)
                tokens.append(sentence_tokens)
                # tokens.append(sentence)
            # print("tokens", tokens)
            anns = []
            while (1):
                line = ner_file.readline()
                if line == "\n":
                    # print("*************")
                    break
                ann = line.strip().split("\t")
                # ann[1] = int(ann[1])
                # ann[2] = int(ann[2])
                ann[1] = int(float(ann[1]))
                ann[2] = int(float(ann[2]))
                ann[-1], ann[-2] = ann[-2], ann[-1]
                # ann = process(ann)
                anns.append(ann[1:])
            anns = sorted(anns, key=lambda x: (x[0], x[1]))
            # print("==============")
            # print("anns", anns)
            sent_offset = get_sent_offset(sentences, text)
            # print("sent_offset", sent_offset)
            token_offset = get_token_offset(tokens, sentences)
            # print("token_offset", token_offset)
#             anns = clean_anns(anns, sent_offset, text)
            anns = change_tags(anns, sent_offset, token_offset)
            # print("anns", anns)
#             anns = normalize_id(anns, human_genes)
            genes, diseases = seperate_genes_and_diseases(anns)
            # print("genes", genes)
            # print("diseases", diseases)
            ann_tag_ordered = make_tags(anns)
            # print("ann_tag_ordered", ann_tag_ordered)
            word_sequence, fixed_features = generate_sequence(tokens, ann_tag_ordered)
            # print("word_sequence", word_sequence)
            # print("fixed_features", fixed_features)
            word_sequence, fixed_features = Filter_rnn(word_sequence, fixed_features)
            # print("fixed_features", fixed_features)
            sequences, feature, gdas = Generate_data_rnn(genes, diseases, word_sequence, fixed_features)

            gdas = [[pmid] + gda for gda in gdas]
            seq_rnn = [texts_to_sequences(text, word_index) for text in sequences]

            seqs.extend(seq_rnn)
            features.extend(feature)
            gdas_total.extend(gdas)
                
        except (IndexError, ValueError, TypeError) as e:
            # print("*****************1")
            print pmid
            continue

    gdas_df = pd.DataFrame(gdas_total, columns=['pmid', "geneId", "follicleId"])
    
    text_file.close()
    sentence_file.close()
    ner_file.close()

    return gdas_df, seqs, features

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    text_path= in_directory + "texts.txt"
    sentence_path = in_directory + "sentences.txt"
    ner_path = in_directory + "anns.txt"
    with open("model/word_index") as fp:
        word_index = cPickle.load(fp)
    
    gdas, x_word_seq, x_feature = load_documents(text_path, sentence_path, ner_path, word_index)  
    label_dir = in_directory +"labels.csv"
    if os.path.exists(label_dir):
        pos_labels = pd.read_csv(label_dir)
        pos_labels.pmid = pos_labels.pmid.astype(str)
        pos_labels.geneId = pos_labels.geneId.astype(str)
        gdas = pd.merge(gdas, pos_labels, on=['pmid', "geneId", "follicleId"], how="left")

        gdas = gdas.fillna(0)

        y = gdas.label.values

        with open(out_directory + '/y', 'wb') as fp:
            cPickle.dump(y, fp)