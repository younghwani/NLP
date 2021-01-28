# -*- coding: utf-8 -*-

# EVN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 단어 긍정(2), 중립(1), 부정(0) 분류 프로젝트
# 데이터
sentences = [
    '나는 오늘 기분이 좋아',
    '나는 오늘 우울해'
]

# 출력 정답
labels = [[1, 1, 1, 2],
          [1, 1, 0]]  # 긍정(2), 중립(1), 부정(0)

# 정답 dic
id_to_label = {0: '부정', 1: '중립', 2: '긍정'}

# Vocabulary
# 각 문장을 띄어쓰기 단위로 분할
words = []
for sentence in sentences:
    words.extend(sentence.split())

# 중복 단어 제거
words = list(dict.fromkeys(words))

# 각 단어별 고유한 번호 부여
word_to_id = {'[PAD]': 0, '[UNK]': 1}
for word in words:
    word_to_id[word] = len(word_to_id)

# 각 숫자별 단어 부여
id_to_word = {_id:word for word, _id in word_to_id.items()}
print('Id to Word : \n', id_to_word)

# 학습용 입력 데이터 생성
train_inputs = []
for sentence in sentences:
    train_inputs.append([word_to_id[word] for word in sentence.split()])

# train label은 labels 사용
train_labels = labels

# 문장의 길이를 모두 동일하게 변경 (최대길이 4)
for row in train_inputs:
    row += [0] * (4 - len(row))

# train inputs을 numpy array로 변환
train_inputs = np.array(train_inputs)

# 정답 길이를 모두 동일하게 변경 (최대길이 4)
for row in train_labels:
    row += [1] * (4 - len(row))

# 학습용 정답을 numpy array로 변환
train_labels = np.array(train_labels)

# 모델링
# 입력 단어를 vector로 변환
embedding = tf.keras.layers.Embedding(len(word_to_id), 8)
hidden = embedding(train_inputs)
print('After Embedding : \n', hidden)

# embedding weight
weight = embedding.get_weights()[0]
print('weight.shape : ', weight.shape)

# numpy를 이용해서 직접 조회 (두 결과값 비교)
# print(weight[train_inputs], hidden)

# 단어의 vector를 이용해서 긍정(2), 부정(0), 중립(1) 확률값 예측
linear = tf.keras.layers.Dense(3, activation=tf.nn.softmax)
outputs = linear(hidden)
print('After linear : \n', outputs)

# dense의 wieght, bias
weight, bias = linear.get_weights()
# print(weight, bias)

# numpy를 이용한 xW + b
logits = np.matmul(hidden, weight) + bias
# print(logits)

# softmax 계산을 위한 준비 exp(x') / sum(exp(x))
numerator = np.exp(logits)
denominator = np.sum(numerator, axis=2, keepdims=True)
# print(numerator, denominator)

# 두 결과값 비교
probs = numerator / denominator
print('probs : ', probs, '\noutputs : ', outputs)

def build_model(n_vocab, d_model, n_seq, n_out):
    """
    동작만 하는 간단한 모델
    :param n_vocab: vocabulary 단어 수
    :param d_model: 단어를 의미하는 벡터의 차원 수
    :param n_seq: 문장길이 (단어 수)
    :param n_out: 예측할 class 개수
    """
    inputs = tf.keras.layers.Input((n_seq,))  # (bs, n_seq)
    embedding = tf.keras.layers.Embedding(n_vocab, d_model) # 입력 단어를 vector로 변환
    hidden = embedding(inputs)  # (bs, n_seq, d_model)
    hidden = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax)(hidden) # 단어의 vector를 이용해서 정답 확률값 예측 (bs, n_seq, n_out)
    model = tf.keras.Model(inputs=inputs, outputs=hidden) # 학습할 모델 선언
    return model

# 모델 생성
model = build_model(len(word_to_id), 8, 4, 3)
# 모델 내용 출력
print(model.summary())

# 학습
# 모델 loss, optimizer, metric 정의
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
history = model.fit(train_inputs, train_labels, epochs=100, batch_size=16)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 모델 평가
model.evaluate(train_inputs, train_labels)

# 예측
# 추론할 입력
string = '나는 기분이 우울해'

# 입력을 숫자로 변경
infer_input = [word_to_id[word] for word in string.split()]
# print(infer_input)

# 문장의 길이를 모두 동일하게 변경 (최대길이 4)
infer_input += [0] * (4 - len(infer_input))
# print(infer_input)

# numpy array 변환 (batch size 1 추가)
infer_inputs = np.array([infer_input])
# print(infer_inputs)

# 긍정/부정 추론
y_preds = model.predict(infer_inputs)
print('y_preds : ', y_preds)

# 확률의 max 값을 추론 값으로 결정
y_pred_class = np.argmax(y_preds, axis=2)
print('y_pred_class : ', y_pred_class)

# 각 예측 값에 대한 label string
for row in y_pred_class:
    for val in row:
        print(val, ':', id_to_label[val])
