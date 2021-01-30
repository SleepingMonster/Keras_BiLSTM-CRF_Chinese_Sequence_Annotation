#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import pandas as pd
import gensim
from gensim.models import KeyedVectors
import tensorflow
import keras
import pickle
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Bidirectional, LSTM, Masking, Dense, Input, TimeDistributed, Activation, Lambda, Dropout
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras import backend as K
from tqdm import tqdm
from keras import optimizers


# In[4]:


# 超参量
SAVE_PATH = 'config.pkl'
BiRNN_UNITS = 200
BATCH_SIZE = 16  # 32、64
EMBED_DIM=300
EPOCHS = 5


# In[5]:


# train_set = open('msrseg/msr_training.utf8', 'r', encoding='utf-8')
train_set = pd.read_csv('msrseg/msr_training.utf8', encoding= 'utf8', header=None)  # 不把第一行作为列属性，且pd读出来就是数据帧，就是字符串
test_set = pd.read_csv('msrseg/msr_test_gold.utf8', encoding='utf8', header=None)
print(train_set.head())
print(test_set.head())


# In[6]:


# 将句子转换成字序列
def get_char(sentence):
    char_list = []
    sentence = ''.join(sentence.split('  ')) #去掉空格
    for i in sentence:
        char_list.append(i)
    return char_list


# In[7]:


#将句子转成BMES序列
def get_label(sentence):
    result = []
    word_list = sentence.split('  ')  #两个空格来分隔一个词
    for i in range(len(word_list)):
        if len(word_list[i]) == 1:
            result.append('S')
        elif len(word_list[i]) == 2:
            result.append('B')
            result.append('E')
        else:
            temp = len(word_list[i]) - 2
            result.append('B')
            result.extend('M'*temp)
            result.append('E')
    return result


# In[8]:


def read_file(file):
    char, content, label = [], [], []
    maxlen = 0

    for i in range(len(file)):  # 记得加range！！
        line = file.loc[i,0]   # 用loc来访问dataframe
        line = line.strip('\n') #去掉换行符
        line = line.strip(' ')  #去掉开头和结尾的空格
        
        char_list = get_char(line)        #获得字列表
        label_list = get_label(line)      # 获得标签列表
        maxlen = max(maxlen, len(char_list))
        if len(char_list)!=len(label_list):
            continue   # 由于数据集的问题，所以要删掉有问题的样本（在训练集中有26个样本；测试集中无）
        char.extend(char_list)            #每一个单元是1个字
        content.append(char_list)         # 每一个单元是一行里面的各个字（分好）
        label.append(label_list)          #每一个单元是一行里面打好标签的结果（含标点）
    return char, content, label, maxlen  #word是单列表，content和label是双层列表


# In[9]:


# process data: padding
def process_data(char_list, label_list, vocab, chunk_tags, MAXLEN):
    # idx2vocab = {idx: char for idx, char in enumerate(vocab)}
    vocab2idx = {char: idx for idx, char in enumerate(vocab)}
    # get every char of every word, map to idx in vocab, set to <UNK> if not in vocab
    x = [[vocab2idx.get(char, 1) for char in s] for s in char_list]
    # map label to idx
    y_chunk = [[chunk_tags.index(label) for label in s] for s in label_list]
    # padding of x, default is 0(symbolizes <PAD>). padding includes:over->cutoff, less->padding. default: left_padding
    x = pad_sequences(x, maxlen=MAXLEN, value=0)
    # padding of y_chunk
    y_chunk = pad_sequences(y_chunk, maxlen=MAXLEN, value=-1)
    # one_hot:
    y_chunk = to_categorical(y_chunk, len(chunk_tags))
    # y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    return x, y_chunk


# In[10]:


def load_data():
    chunk_tags = ['S','B','M','E']
    train_char, train_content, train_label, _ = read_file(train_set)
    test_char, test_content, test_label, maxlen = read_file(test_set)
    
    vocab = list(set(train_char + test_char))   # 合并，构成大词表
    special_chars = ['<PAD>', '<UNK>']   #特殊词表示：PAD表示padding，UNK表示词表中没有
    vocab = special_chars + vocab
    
    # save initial config data
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump((train_char, chunk_tags), f)
    
    # process data: padding
    print('maxlen is %d' % maxlen)
    train_x, train_y = process_data(train_content, train_label, vocab, chunk_tags, maxlen)
    test_x, test_y = process_data(test_content, test_label, vocab, chunk_tags, maxlen)
    return train_x, train_y, test_x, test_y, vocab, chunk_tags, maxlen, test_content


# In[11]:


word2vec_model_path = 'sgns.context.word-character.char1-1.bz2'  #词向量位置
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore')


# In[12]:


def make_embeddings_matrix(word2vec_model, vocab):
    char2vec_dict = {}    # 字对词向量
    vocab2idx = {char: idx for idx, char in enumerate(vocab)}
    for char, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        char2vec_dict[char] = vector
    embeddings_matrix = np.zeros((len(vocab), EMBED_DIM))# form huge matrix
    for i in tqdm(range(2, len(vocab))):
        char = vocab[i]
        if char in char2vec_dict.keys():    # 如果char在词向量列表中，更新权重；否则，赋值为全0（默认）
            char_vector = char2vec_dict[char]
            embeddings_matrix[i] = char_vector
    return embeddings_matrix


# In[13]:


# K.clear_session()
train_x, train_y, test_x, test_y, vocab, chunk_tags, maxlen, test_content = load_data()
embeddings_matrix = make_embeddings_matrix(word2vec_model, vocab)
# input layer
inputs = Input(shape=(maxlen, ), dtype='int32')
# masking layer 屏蔽层
x = Masking(mask_value=0)(inputs)
# embedding layer: map the word to it's weights(with embedding-matrix)
x = Embedding(len(vocab), EMBED_DIM, weights=[embeddings_matrix], input_length=maxlen, trainable=True)(x)
# Bi-LSTM layer
x = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True))(x)
# Dropout: 正则化，防止过拟合.argument means percentage
x = Dropout(0.5)(x)
# 一维展开，全连接
x = TimeDistributed(Dense(len(chunk_tags)))(x)
# output layer
outputs = CRF(len(chunk_tags))(x)
# model
model = Model(inputs=inputs, outputs=outputs)
# print arguments of each layer
model.summary()
# target_function: includes optimizer, function_type, metrics
# SGD = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
SGD = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
SGDM = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
RMSprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])


# In[ ]:


# train
model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.1)
# test_predict = model.predict(test_x)
score = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
print(score)
model.save_weights('model.h5')


# In[ ]:


# test
model.load_weights('model.h5')
test_predict = model.predict(test_x)


# In[ ]:


test_predict = [[np.argmax(char) for char in sample] for sample in test_predict]  # get the max label_id
test_predict_tag = [[chunk_tags[i] for i in sample ]for sample in test_predict]   # get the label of predic
test_gold = [[np.argmax(char) for char in sample] for sample in test_y]  # get the label_id
test_gold_tag = [[chunk_tags[i] for i in sample] for sample in test_gold]  # get the label of real


# In[ ]:


_, test_content, _, _ = read_file(test_set)
f_sum = 0   # 各个句子的f值之和
for i in range(len(test_predict_tag)):
    correct_word_num = 0  # 分对词数
    predict_word_num = 0  # 预测分词结果的总词数
    gold_word_num = 0     # 实际分词结果总词数
    predict_sample = test_predict_tag[i]
    gold_sample = test_gold_tag[i]
    s_len = len(test_content[i])   # the real length of the sentence
    flag = False   # true: inside a word; false: outside a word
    for j in range(len(predict_sample) - s_len, len(predict_sample)):
        if gold_sample[j] == 'S' or gold_sample[j] == 'E' or j == len(predict_sample) - 1:   # update gold_word_num
            gold_word_num += 1
        if predict_sample[j] == 'S' or predict_sample[j] == 'E' or j == len(predict_sample) - 1:   # update predict_word_num
            predict_word_num += 1
        if gold_sample[j] != predict_sample[j]:
            flag = False
            continue
        elif gold_sample[j] == predict_sample[j] and (gold_sample[j] == 'S' or (gold_sample[j] == 'E' and flag is True)):
            correct_word_num += 1
            flag = False
        elif gold_sample[j] == predict_sample[j] and gold_sample[j] == 'B':
            flag = True   # inside the word: start
    precision = float(correct_word_num) /float( predict_word_num)
    recall = float(correct_word_num) / float(gold_word_num)
    if precision == 0 and recall == 0: f1 = 0
    else: f1 = 2 * precision * recall / (precision + recall)
    f_sum += f1
print(f_sum / len(test_predict_tag))


# In[113]:


test_result = []
vocab2idx = {char: idx for idx, char in enumerate(vocab)}
_, test_content, _, _ = read_file(test_set)
for i in range(len(test_predict)):
# for i in range(1):
    sentence = ''
    s_len = len(test_content[i])
    sample = test_predict_tag[i]
    for j in range(s_len):
        idx = len(sample)- s_len + j
        if sample[idx]=='B' or sample[idx]=='M' or j==s_len-1:
            sentence = sentence + test_content[i][j]
        else:
            sentence = sentence + test_content[i][j]
            sentence = sentence + '  '
    test_result.append(sentence)
print(test_result[:5])


# In[114]:


file = open('msrseg/msr_test_predict.txt', 'w')
file1 = open('msrseg/msr_test_predict.utf8', 'w', encoding='utf-8')   # 当不是对应位置的参数时，要加上参数名！
for i in range(len(test_result)):
    file.write(test_result[i])
    file1.write(test_result[i])
    file.write('\n')
    file1.write('\n')
file.close()
file1.close()


# In[115]:


test_predict = pd.read_csv('msrseg/msr_test_predict_10.utf8', encoding= 'utf8', header=None)  # 不把第一行作为列属性，且pd读出来就是数据帧，就是字符串
print(test_predict.head)

