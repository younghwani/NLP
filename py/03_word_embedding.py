# -*- coding: utf-8 -*-
# # Evn
import os
import random
import shutil
import json
import zipfile
import math
import copy
import collections
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sentencepiece as spm
import konlpy # 행태소분석기
import gensim
import gensim.downloader as api
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA

# random seed initialize
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

okt = konlpy.tag.Okt()
print(okt.morphs("아버지가방에들어가신다"))

# 네이버 영화 리뷰 데이터
nsmc_data = pd.read_csv("Data/nlp/nsmc/ratings.txt", header=0, delimiter='\t', quoting=3)
print(f"전체 데이터의 개수: {len(nsmc_data)}")
print(nsmc_data.head(10))

# null 제거
nsmc_data = nsmc_data.dropna()
print(f"null 제거 후 데이터의 개수: {len(nsmc_data)}")
print(nsmc_data.head(10))

# 한글 이외의 문자 제거
nsmc_data['document'] = nsmc_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
nsmc_data.dropna()
print(f"한글 아닌 문자 제거 후 데이터의 개수: {len(nsmc_data)}")
print(nsmc_data.head(10))

# 불용어 정의 (빈도가 너무 많은 단어는 학습에서 제외 함)
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# okt 형태소 분석기를 이용해 형태소 단위로 분할 (이때 불용어 제거)
tokens = []
for i, document in enumerate(tqdm(nsmc_data['document'], total=len(nsmc_data))):
    line = []
    line = okt.morphs(document)
    line = [word for word in line if not word in stopwords]
    tokens.append(line)

print(len(tokens))
print(tokens[:10])

# gensim 학습
word2vec_100 = gensim.models.Word2Vec(sentences=tokens, size=100, window=5, min_count=5)

# test
words = list(word2vec_100.wv.vocab)
print(len(words), words[:100])
similar = word2vec_100.wv.most_similar("영화")
print(similar)
similar = word2vec_100.wv.most_similar("최민수")
print(similar)
similar = word2vec_100.wv.most_similar("장동건")
print(similar)
# 설경구 - 송윤아 + 고소영
result = word2vec_100.wv.most_similar(positive=['고소영', '설경구'], negative=['송윤아'])
print(result)
# '아이언'을 입력하면 아이언맨이 나왔으면, 혹은 아이언맨과 관련한 사항이 나왔으면 했으나 특성 상 두 단어 사이의 연관성이 없는 것을 확인할 수 있었습니다.
print(word2vec_100.wv.most_similar("아이언"))
print(word2vec_100.wv.most_similar("아이언맨"))

'''
print(word2vec_100.wv.most_similar("동막골"))
word2vec_100.wv.most_similar("미이라")
result = word2vec_100.wv.most_similar(positive=['아이언맨', '캡틴'], negative=['아메리카'])
result
word2vec_100.wv.most_similar_to_given('송강호', ['라미란', '신민아', '신동엽', '신세경', '신발', '옥상', '아파트', '기생충', '건달'])
word2vec_100.wv.most_similar('학생')
word2vec_100.wv.most_similar('일본')
word2vec_100.wv.most_similar(positive=['한국', '프랑스', '고전'])
word2vec_100.wv.most_similar(positive=['한국', '인도'], negative=['중국'])
word2vec_100.wv.most_similar(positive=['중국'], negative=['미국'])
word2vec_100.wv.most_similar(positive=['삼류'], negative=['명작'])
word2vec_100.wv.most_similar(positive=['명품', '명작'], negative=['삼류', '똥', '망'])
word2vec_100.wv.most_similar(positive=['액션'], negative=['멜로'])
word2vec_100.wv.most_similar(positive=['멜로'], negative=['액션'])
'''