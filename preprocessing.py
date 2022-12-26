import json

import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pad_packed_sequence
import torch



def convert_examples_to_features(x, y,
                                 max_seq_length,
                                 tokenizer,
                                 jlbert_tokenizer=None,
                                 debug=False):
    y=torch.tensor(y, dtype=torch.long)
    features = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': y
    }
    for pairs in x:
        tokens = [tokenizer.cls_token]
        token_type_ids = []
        for i, sent in enumerate(pairs):
            if jlbert_tokenizer!=None:
                sent=jlbert_tokenizer(sent)
            word_tokens = tokenizer.tokenize(sent)
            tokens.extend(word_tokens)
            tokens += [tokenizer.sep_token]
            len_sent = len(word_tokens) + 1
            token_type_ids += [i] * len_sent

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        features['input_ids'].append(torch.tensor(input_ids, dtype=torch.int32))
        features['attention_mask'].append(torch.tensor(attention_mask, dtype=torch.int32))
        features['token_type_ids'].append(torch.tensor(token_type_ids, dtype=torch.int32))
    
    if debug:
        # print(features['input_ids'])
        array=list(map(len,features["input_ids"]))
        print(array)
        print(len(array))
        return array

    for name in ['input_ids', 'attention_mask', 'token_type_ids']:
        # 最後に埋めている
        # features[name] = pad_sequences(features[name], padding='post', maxlen=max_seq_length)
        temp = pad_sequence(features[name], batch_first=True) # tensorに変換。最大長に末尾0埋めで合わせる。
        # print(temp.shape)
        temp=temp[:,:max_seq_length] # 長ければぶった斬る
        packed = pack_sequence(temp,enforce_sorted=False)
        out,_=pad_packed_sequence(packed, batch_first=True, total_length=max_seq_length) # 短ければ0を追加する
        features[name]=out
        # print(out.shape)
    x = features

    return x, y

def convert_examples_to_multi_bert_features(x, y,
                                 max_seq_length,
                                 tokenizer,
                                 jlbert_tokenizer=None,
                                 debug=False):
    y=torch.tensor(y, dtype=torch.long)
    features = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': y
    }
    for pairs in x:
        input_set=[]
        attention_mask_set=[]
        token_type_set=[]
        for i, sent in enumerate(pairs):
            tokens = [tokenizer.cls_token]
            token_type_ids = []
            if jlbert_tokenizer!=None:
                sent=jlbert_tokenizer(sent)
            word_tokens = tokenizer.tokenize(sent)
            tokens.extend(word_tokens)
            tokens += [tokenizer.sep_token]
            len_sent = len(word_tokens) + 1
            token_type_ids += [i] * len_sent

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            input_set.append(torch.tensor(input_ids,dtype=torch.int32))
            attention_mask_set.append(torch.tensor(attention_mask,dtype=torch.int32))
            token_type_set.append(torch.tensor(token_type_ids,dtype=torch.int32))

        features['input_ids'].append(input_set)
        features['attention_mask'].append(attention_mask_set)
        features['token_type_ids'].append(token_type_set)

    features['input_ids']=[list(x) for x in zip(*features['input_ids'])] # shape(sentences,pair,len) to (pair,sentences,len)
    features['attention_mask']=[list(x) for x in zip(*features['attention_mask'])]
    features['token_type_ids']=[list(x) for x in zip(*features['token_type_ids'])]
    
    # print("debug",len(features["attention_mask"]),len(features["attention_mask"][0]))
    
    if debug:
        # print(features['input_ids'])
        array=list(map(len,features["input_ids"]))
        print(array)
        print(len(array))
        return array

    for name in ['input_ids', 'attention_mask', 'token_type_ids']:
        # print("\n\ndebug",len(features[name]))
        for i in range(len(features[name])):
            # print("\n\ndebug",features[name][i])
            # features[name][i] = pad_sequences(features[name][i], padding='post', maxlen=max_seq_length)

            temp = pad_sequence(features[name][i], batch_first=True) # tensorに変換。最大長に末尾0埋めで合わせる。
            # print(temp.shape)
            temp=temp[:,:max_seq_length] # 長ければぶった斬る
            packed = pack_sequence(temp,enforce_sorted=False)
            out,_=pad_packed_sequence(packed, batch_first=True, total_length=max_seq_length) # 短ければ0を追加する
            features[name][i]=out

    inp1,inp2=features["input_ids"] # inp 1 shape ... (SeqLen,512)
    att1,att2=features["attention_mask"]
    typ1,typ2=features["token_type_ids"]
    

    # x = (inp1,att1,typ1,inp2,att2,typ2)
    x={"id_p":inp1,"att_p":att1,"typ_p":typ1,"id_h":inp2,"att_h":att2,"typ_h":typ2,"labels":y}
    # text1,text2=list(zip(*x))
    # x["text1"]=text1
    # x["text2"]=text2
    # print("debughoge",x[0].shape)
    print("debuhg",y.shape)
    # x["labels"]=y
    #print("\n\nmulti converter")
    #print(y)
    return x, y


# def convert_examples_to_multi_bert_features(x, y,
#                                  max_seq_length,
#                                  tokenizer,debug=False):
#     features = {
#         'input_ids': [],
#         'attention_mask': [],
#         'token_type_ids': [],
#         'label_ids': np.asarray(y)
#     }
#     for pairs in x:
#         input_set=[]
#         attention_mask_set=[]
#         token_type_set=[]
#         for i, sent in enumerate(pairs):
#             tokens = [tokenizer.cls_token]
#             token_type_ids = []
#             word_tokens = tokenizer.tokenize(sent)
#             tokens.extend(word_tokens)
#             tokens += [tokenizer.sep_token]
#             len_sent = len(word_tokens) + 1
#             token_type_ids += [i] * len_sent

#             input_ids = tokenizer.convert_tokens_to_ids(tokens)
#             attention_mask = [1] * len(input_ids)

#             input_set.append(input_ids)
#             attention_mask_set.append(attention_mask)
#             token_type_set.append(token_type_ids)

#         features['input_ids'].append(input_set)
#         features['attention_mask'].append(attention_mask_set)
#         features['token_type_ids'].append(token_type_set)

#     features['input_ids']=[list(x) for x in zip(*features['input_ids'])] # shape(sentences,pair) to (pair,sentences)
#     features['attention_mask']=[list(x) for x in zip(*features['attention_mask'])]
#     features['token_type_ids']=[list(x) for x in zip(*features['token_type_ids'])]
    
#     # print("debug",len(features["attention_mask"]),len(features["attention_mask"][0]))
    
#     if debug:
#         # print(features['input_ids'])
#         array=list(map(len,features["input_ids"]))
#         print(array)
#         print(len(array))
#         return array

#     for name in ['input_ids', 'attention_mask', 'token_type_ids']:
#         # print("\n\ndebug",len(features[name]))
#         for i in range(len(features[name])):
#             # print("\n\ndebug",features[name][i])
#             features[name][i] = pad_sequences(features[name][i], padding='post', maxlen=max_seq_length)
    
#     features['input_ids']=[list(x) for x in zip(*features['input_ids'])] # shape (pair,sentences) to (sentences,pair)
#     features['attention_mask']=[list(x) for x in zip(*features['attention_mask'])]
#     features['token_type_ids']=[list(x) for x in zip(*features['token_type_ids'])]
#     x = (np.array(features['input_ids'], dtype='int32'),
#          np.array(features['attention_mask'], dtype='int32'),
#          np.array(features['token_type_ids'], dtype='int32'))
#     print("debughoge",x[0].shape)
#     try:
#         y = np.array(features['label_ids'], dtype='int32')
#     except BaseException:
#         y = np.array(features['label_ids'])
#     return x, y