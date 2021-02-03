# -*- coding: utf-8 -*-

# Evn
import os
import random
import shutil
import json
import zipfile
import math
import copy
import collections
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm.notebook import tqdm

# random seed initialize
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# data dir
data_dir = 'Data/nlp'
print(os.listdir(data_dir))

# korean wiki dir
kowiki_dir = os.path.join(data_dir, 'kowiki')
if not os.path.exists(kowiki_dir):
    os.makedirs(kowiki_dir)
print(os.listdir(kowiki_dir))

# Vocabulary & config
# vocab loading
# Use kowiki data 
vocab = spm.SentencePieceProcessor()
vocab.load(os.path.join(data_dir, 'ko_32000.model'))

n_vocab = len(vocab)  # number of vocabulary
n_seq = 256  # number of sequence
d_model = 256  # dimension of model

# 모델링
def build_model(n_vocab, d_model, n_seq):
    """
    language_model
    :param n_vocab: vocabulary 단어 수
    :param d_model: 단어를 의미하는 벡터의 차원 수
    :param n_seq: 문장 길이 (단어 수)
    """
    inputs = tf.keras.layers.Input((n_seq,))  # (bs, n_seq)
    # 입력 단어를 vector로 변환
    embedding = tf.keras.layers.Embedding(n_vocab, d_model)
    hidden = embedding(inputs)  # (bs, n_seq, d_model)
    # LSTM
    lstm = tf.keras.layers.LSTM(units=d_model * 2, return_sequences=True)
    hidden = lstm(hidden)  # (bs, n_seq, d_model * 2)
    # 다음단어 확률 분포
    dense = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)
    outputs = dense(hidden)
    # 학습할 모델 선언
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Preprocessing
# 파일 내용 확인
with zipfile.ZipFile(os.path.join(kowiki_dir, 'kowiki.txt.zip')) as z:
    with z.open('kowiki.txt') as f:
        for i, line in enumerate(f):
            line = line.decode('utf-8').strip()
            print(line)
            if i >= 100:
                break

# 파일 내용 확인 (주제단위)
with zipfile.ZipFile(os.path.join(kowiki_dir, 'kowiki.txt.zip')) as z:
    with z.open('kowiki.txt') as f:
        doc = []
        for i, line in enumerate(f):
            line = line.decode('utf-8').strip()
            if len(line) == 0:
                if len(doc) > 0:
                    break
            else:
                doc.append(line)
print(doc)

def create_train_instance(vocab, n_seq, doc):
    """
    create train instance
    :param vocab: vocabulary object
    :param n_seq: sequece number
    :param doc: wiki document
    :return: train instance list
    """
    n_max = n_seq - 1
    instance_list = []

    chunk = []
    chunk_len = 0
    for i, line in enumerate(doc):
        tokens = vocab.encode_as_pieces(line)
        chunk.append(tokens)
        chunk_len += len(tokens)
        if n_max <= chunk_len or i >= len(doc) -1:
            # print()
            # print(chunk_len, chunk)
            instance = []
            for tokens in chunk:
                instance.extend(tokens)
            # print(len(instance), instance)
            instance = instance[:n_max]
            # print(len(instance), instance)
            instance_list.append(instance)
            chunk = []
            chunk_len = 0

    return instance_list

# instance 동작 확인
instance_list = create_train_instance(vocab, n_seq, doc)
for instance in instance_list:
    print(len(instance), instance)

# instance를 json 형태로 저장하는 함수
def save_instance(vocab, n_seq, doc, o_f):
    instance_list = create_train_instance(vocab, n_seq, doc)
    for instance in instance_list:
        o_f.write(json.dumps({'token': instance}, ensure_ascii=False))
        o_f.write('\n')

# 전체 문서에 대한 instance 생성
with open(os.path.join(kowiki_dir, 'kowiki_lm.json'), 'w') as o_f:
    with zipfile.ZipFile(os.path.join(kowiki_dir, 'kowiki.txt.zip')) as z:
        with z.open('kowiki.txt') as f:
            doc = []
            for i, line in enumerate(tqdm(f)):
                line = line.decode('utf-8').strip()
                if len(line) == 0:
                    if len(doc) > 0:
                        save_instance(vocab, n_seq, doc, o_f)
                        doc = []
                else:
                    doc.append(line)
            if len(doc) > 0:
                save_instance(vocab, n_seq, doc, o_f)

# 파일 라인수 확인
n_line = 0
with open(os.path.join(kowiki_dir, 'kowiki_lm.json')) as f:
    for line in f:
        n_line += 1
        if n_line <= 10:
            print(line)
print(n_line)


## Data
def load_data(vocab, n_seq):
    """
    Language Model 학습 데이터 생성
    :param vocab: vocabulary object
    :param n_seq: number of sequence
    :return inputs_1: input data 1
    :return inputs_2: input data 2
    :return labels: label data
    """
    # line 수 조회
    n_line = 0
    with open(os.path.join(kowiki_dir, 'kowiki_lm.json')) as f:
        for line in f:
            n_line += 1
    #print(n_line) # n_line 값 : 778381
    # 최대 100,000개 데이터
    n_data = min(n_line, 100000)
    # 빈 데이터 생성
    inputs = np.zeros((n_data, n_seq)).astype(np.int32)
    labels = np.zeros((n_data, n_seq)).astype(np.int32)

    with open(os.path.join(kowiki_dir, 'kowiki_lm.json')) as f:
        for i, line in enumerate(tqdm(f, total=n_data)):
            if i >= n_data:
                break
            data = json.loads(line)
            token_id = [vocab.piece_to_id(p) for p in data['token']]
            # input id
            input_id = [vocab.bos_id()] + token_id
            input_id += [0] * (n_seq - len(input_id))
            # label id
            label_id = token_id + [vocab.eos_id()]
            label_id += [0] * (n_seq - len(label_id))
            # 값 저장
            inputs[i] = input_id
            labels[i] = label_id

    return inputs, labels

# train data 생성
train_inputs, train_labels = load_data(vocab, n_seq)
print(train_inputs, train_labels)

# Loss & Acc
def lm_loss(y_true, y_pred):
    """
    pad 부분을 제외하고 loss를 계산하는 함수
    :param y_true: 정답
    :param y_pred: 예측 값
    :retrun loss: pad 부분이 제외된 loss 값
    """
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)
    loss *= mask
    return loss

def lm_acc(y_true, y_pred):
    """
    pad 부분을 제외하고 accuracy를 계산하는 함수
    :param y_true: 정답
    :param y_pred: 예측 값
    :retrun loss: pad 부분이 제외된 accuracy 값
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    matches = tf.cast(tf.equal(y_true, y_pred_class), tf.float32)
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)
    matches *= mask
    accuracy = tf.reduce_sum(matches) / tf.maximum(tf.reduce_sum(mask), 1)
    return accuracy

# 학습
# 모델 생성
model = build_model(len(vocab), d_model, n_seq)
# 모델 내용 출력
model.summary()

# 모델 loss, optimizer, metric 정의
model.compile(loss=lm_loss, optimizer='adam', metrics=[lm_acc])

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='lm_acc', patience=5)
# save weights callback
save_weights = tf.keras.callbacks.ModelCheckpoint(os.path.join(kowiki_dir, 'lm.hdf5'),
                                                  monitor='lm_acc',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  mode="max",
                                                  save_freq="epoch",
                                                  save_weights_only=True)
# csv logger
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(kowiki_dir, 'lm.csv'))

# 모델 학습
history = model.fit(train_inputs,
                    train_labels, 
                    epochs=2,
                    batch_size=64,  # oom이 발생하면 값을 줄여주세요.
                    callbacks=[early_stopping, save_weights, csv_logger])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['lm_acc'], 'g-', label='acc')
plt.xlabel('Epoch')
plt.legend()

plt.show()