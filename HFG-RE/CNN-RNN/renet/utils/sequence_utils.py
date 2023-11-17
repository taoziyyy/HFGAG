
def make_tags(tags):
    tag_duplicates = []
    last_sent = -1
    last_start = -1
    last_end = -1
    last_type = ''
    last_Id = -1
    last_mention = ''
    for i, tag in enumerate(tags):
        sent, start_offset, end_offset, tag_type, mention, Id = tag[0], tag[1], tag[2], tag[-1], str(tag[3]).lower(), tag[-2]
        if sent == last_sent and end_offset == last_end and start_offset == last_start and mention == last_mention:
#             start_tokens = [token for token in abstract_tokens_normal_clean[pubmed][sent][start_offset:end_offset+1]]
#             end_tokens = abstract_tokens[pubmed][sent][last_start:last_end+1]
            if last_tag != '' and last_tag_type != tag_type:
                tag[-1] = 'Gene-Follicle'
                if tag_type == 'Gene':
                    tag[-2] = str(last_Id) + '>-<' + str(Id)
                else:
                    tag[-2] = str(Id) + '>-<' + str(last_Id)
                tag_duplicates.append(i-1)
        last_sent = sent
        last_start = start_offset
        last_end = end_offset
        last_tag_type = tag_type
        last_Id = Id
        last_tag = tag
        last_mention = mention
    tags = [tags[i] for i in range(len(tags)) if i not in tag_duplicates]
    
    new_tags = []
    last_sent = -1
    last_start = -1
    last_end = -1
    last_tag = ''
    last_Id = -1
    last_tag_name = ''
    for tag in tags:
        sent, start_offset, end_offset, mention, Id, tag_type, tag_name = \
            tag[0], tag[1], tag[2], str(tag[3]).lower(), tag[4], tag[-1], '<' + tag[-2] + '>'
        if sent == last_sent and start_offset <= last_end and last_tag != '':
            if last_tag_name == tag_name:
                new_tag_name = tag_name
                new_tag_type = tag_type
            elif type(last_tag_name) == list:
                new_tag_name = last_tag_name + [tag_name]
                new_tag_type = last_tag_type + [tag_type]
            else:
                new_tag_name = [last_tag_name, tag_name]
                new_tag_type = [last_tag_type, tag_type]
            new_tag = [sent, last_start, end_offset, mention, new_tag_name, new_tag_type]

            del new_tags[-1]
            new_tags.append(new_tag)
        else:
            new_tags.append(tag[:-2] + [tag_name] + [tag[-1]])

        last_tag = new_tags[-1]
        last_sent, last_start, last_end, last_mention, last_Id, last_tag_type, last_tag_name = \
            last_tag[0], last_tag[1], last_tag[2], str(last_tag[3]).lower(), last_tag[4], last_tag[-1], last_tag[-2]
    
    return new_tags

def generate_sequence(sentences, tags):

    new_sentences = []
    sent_features = []
    tag_no = 0
    for sentence_no in range(len(sentences)):
        sentence = sentences[sentence_no]
        if tag_no == len(tags):
            new_sentences.append(sentence)
            sent_features.append([[0, 0, 0, 0]] * len(sentence))
            continue
        tag = tags[tag_no]
        if tag[0] == sentence_no:
            last_end = 0
            new_sentence = []
            sent_feature = []
            while tag[0] == sentence_no:
                start = tag[1]
                end = tag[2] + 1
                if type(tag[-2]) == list:
                    tag_name = tag[-2]
                    feature = [One_hot_feature(tag_type) for tag_type in tag[-1]]
                else:
                    tag_name = [tag[-2]]
                    feature = [One_hot_feature(tag[-1])]
                new_sentence += sentence[last_end:start] + tag_name
                sent_feature += [[0, 0, 0, 0]] * len(sentence[last_end:start]) + feature
                tag_no += 1
                last_end = end
                if  tag_no < len(tags):
                    tag = tags[tag_no]
                else:
                    break
            new_sentence += sentence[end:]
            sent_feature += [[0, 0, 0, 0]] * len(sentence[end:])

            new_sentences.append(new_sentence)
            sent_features.append(sent_feature)
        else:
            new_sentences.append(sentence)
            sent_features.append([[0, 0, 0, 0]] * len(sentence))
    return new_sentences, sent_features

def Filter_rnn(word_seq, fixed_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    
    word_seq_filter = []
    fixed_features_filter = []
    for sentence_no in range(len(word_seq)):
        sent_word_seq = word_seq[sentence_no]
        sent_features = fixed_features[sentence_no]
        sent_word_seq_filter = []
        sent_features_filter = []
        for token_no in range(len(sent_word_seq)):
            token = sent_word_seq[token_no]
            feature = sent_features[token_no]
            if token not in filters:
                if sum(feature[:4]) == 0:
                    sent_word_seq_filter.append(token.lower())
                else:
                    sent_word_seq_filter.append(token)
                sent_features_filter.append(feature[:4] + feature[-2:])
        word_seq_filter.append(sent_word_seq_filter)
        fixed_features_filter.append(sent_features_filter)
    return word_seq_filter, fixed_features_filter

def One_hot_feature(tag_type):
    if tag_type == "Follicle":
        return [1, 0, 0, 0]
    elif tag_type == 'Gene':
        return [0, 1, 0, 0]
    elif tag_type == 'Gene-Follicle':
        return [1, 1, 0, 0]
    
def texts_to_sequences_generator(texts, word_index):
    """Transforms each text in texts in a sequence of integers.
    Only top "num_words" most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    # Arguments
        texts: A list of texts (strings).
    # Yields
        Yields individual sequences.
    """
#     num_words = self.num_words
    for seq in texts:
        vect = []
        for w in seq:
            i = word_index.get(w)
            if i is not None:
                vect.append(i)
            else:
                vect.append(word_index['UUUNKKK'])
        yield vect

def texts_to_sequences(texts, word_index):
        """Transforms each text in texts in a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Returns
            A list of sequences.
        """
        res = []
        for vect in texts_to_sequences_generator(texts, word_index):
            res.append(vect)
        return res
    
def Generate_data_rnn(genes, follicles, word_sequences, fixed_features):
    index = 0
    df_fixed_features = []
    df_word_sequences = []
    gdas = []
    for geneId in genes.keys():
        for follicleId in follicles.keys():
            find_target_gene = False
            find_target_follicle = False
            target_gene = '<' + geneId + '>'
            target_follicle = '<' + follicleId + '>'
            pubmed_features = []
            for sentence_no in range(len(word_sequences)):
                sentence = word_sequences[sentence_no]
                features = fixed_features[sentence_no]
                new_features = []
                for token_no in range(len(sentence)):
                    feature = features[token_no]
                    token = sentence[token_no]
                    if target_follicle in token.split('-') and target_gene in token.split('-'):
                        find_target_gene = True
                        find_target_follicle = True
                        new_feature = [6]
                        new_features.append(new_feature)
                    elif (target_follicle in token.split('-') and feature[0] == 1):
                        find_target_follicle = True
                        new_feature = [4]
                        new_features.append(new_feature)
                    elif target_gene in token.split('-') and feature[1] == 1:
                        find_target_gene = True
                        new_feature = [5]
                        new_features.append(new_feature)
                    elif (target_follicle not in token.split('-') and feature[0] == 1):
                        non_td = token.split('-')[0].strip('<>')
                        td = target_follicle.strip('<>')
                        new_feature = [feature[0] + 2*feature[1]]
                        new_features.append(new_feature)
                    else:
                        new_features.append([feature[0] + 2*feature[1]])
                pubmed_features.append(new_features)
            if find_target_follicle and find_target_gene:
                df_fixed_features.append(pubmed_features)
                df_word_sequences.append(word_sequences)
                gdas.append([geneId, follicleId])
            
        
    return df_word_sequences, df_fixed_features, gdas
