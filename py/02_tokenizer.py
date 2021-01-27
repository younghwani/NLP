# -*- coding: utf-8 -*-
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

# ENV
# random seed initialize
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# data dir
data_dir = 'Data/nlp'
print('Data/nlp/ list : ', os.listdir(data_dir))

# kowiki dir
kowiki_dir = os.path.join(data_dir, 'kowiki')
if not os.path.exists(kowiki_dir):
    os.makedirs(kowiki_dir)
print('Data/nlp/kowiki/ list : ', os.listdir(kowiki_dir))

# Corpus
file = open(os.path.join(kowiki_dir, 'my_corpus.txt'), 'r', encoding='utf-8')
my_text = file.read()
# 결과 너무 길어서 주석처리, 확인 필요시 주석해제
# print('my_text : \n', my_text)

# 예제 text 
text = """위키백과의 최상위 도메인이 .com이던 시절 ko.wikipedia.com에 구판 미디어위키가 깔렸으나 한글 처리에 문제가 있어 글을 올릴 수도 없는 이름뿐인 곳이었다. 2002년 10월에 새로운 위키 소프트웨어를 쓰면서 한글 처리 문제가 풀리기 시작했지만, 가장 많은 사람이 쓰는 인터넷 익스플로러에서는 인코딩 문제가 여전했다. 이런 이유로 초기에는 도움말을 옮기거나 쓰는 일에 어려움을 겪었다. 이런 어려움이 있었는데도 위키백과 통계로는, 2002년 10월에서 2003년 7월까지 열 달 사이에 글이 13개에서 159개로 늘었고 2003년 7월과 8월 사이에는 한 달 만에 159개에서 348개로 늘어났다. 2003년 9월부터는 인터넷 익스플로러의 인코딩 문제가 사라졌으며, 대한민국 언론에서도 몇 차례 위키백과를 소개하면서 참여자가 점증하리라고 예측했다. 참고로 한국어 위키백과의 최초 문서는 2002년 10월 12일에 등재된 지미 카터 문서이다.
2005년 6월 5일 양자장론 문서 등재를 기점으로 총 등재 문서 수가 1만 개를 돌파하였고 이어 동해 11월에 제1회 정보트러스트 어워드 인터넷 문화 일반 분야에 선정되었다. 2007년 8월 9일에는 한겨레21에서 한국어 위키백과와 위키백과 오프라인 첫 모임을 취재한 기사를 표지 이야기로 다루었다.
2008년 광우병 촛불 시위 때 생긴 신조어인 명박산성이 한국어 위키백과에 등재되고 이 문서의 존치 여부를 두고 갑론을박의 과정이 화제가 되고 각종 매체에도 보도가 되었다. 시위대의 난입과 충돌을 방지하기 위해 거리에 설치되었던 컨테이너 박스를 이명박 정부의 불통으로 풍자를 하기 위해 사용된 이 신조어는 중립성을 지켰는지와 백과사전에 올라올 만한 문서인지가 쟁점이 되었는데 일시적으로 사용된 신조어일 뿐이라는 주장과 이미 여러 매체에서 사용되어 지속성이 보장되었다는 주장 등 논쟁이 벌어졌고 다음 아고라 등지에서 이 항목을 존치하는 방안을 지지하는 의견을 남기기 위해 여러 사람이 새로 가입하는 등 혼란이 빚어졌다. 11월 4일에는 다음커뮤니케이션에서 글로벌 세계 대백과사전을 기증받았으며, 2009년 3월에는 서울특별시로부터 콘텐츠를 기증받았다. 2009년 6월 4일에는 액세스권 등재를 기점으로 10만 개 문서 수를 돌파했다.
2011년 4월 16일에는 대한민국에서의 위키미디어 프로젝트를 지원하는 모임을 결성할 것을 추진하는 논의가 이뤄졌고 이후 창립준비위원회 결성을 거쳐 2014년 10월 19일 창립총회를 개최하였으며, 최종적으로 2015년 11월 4일 사단법인 한국 위키미디어 협회가 결성되어 활동 중에 있다. 2019년 미국 위키미디어재단으로부터 한국 지역 지부(챕터)로 승인을 받았다.
2012년 5월 19일에는 보비 탬블링 등재를 기점으로 총 20만 개 문서가 등재되었고 2015년 1월 5일, Rojo -Tierra- 문서 등재를 기점으로 총 30만 개 문서가 등재되었다. 2017년 10월 21일에는 충청남도 동물위생시험소 문서 등재로 40만 개의 문서까지 등재되었다."""

'''
# Colab에서 작업 시 drive에서 text 파일 가져오려 사용함.
print(os.listdir(kowiki_dir))

shutil.copy(os.path.join(kowiki_dir, 'my_corpus.txt'), './')
os.listdir('./')
'''

def train_sentencepiece(corpus, prefix, vocab_size=32000):
    """
    sentencepiece를 이용해 vocab 학습
    :param corpus: 학습할 말뭉치
    :param prefix: 저장할 vocab 이름
    :param vocab_size: vocab 개수
    """
    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +  # 7은 특수문자 개수
        " --model_type=unigram" +
        " --max_sentence_length=999999" +  # 문장 최대 길이
        " --pad_id=0 --pad_piece=[PAD]" +  # pad token 및 id 지정
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown token 및 id 지정
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence token 및 id 지정
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence token 및 id 지정
        " --user_defined_symbols=[SEP],[CLS],[MASK]")  # 기타 추가 토큰 SEP: 4, CLS: 5, MASK: 6

# vocab 생성
train_sentencepiece(os.path.join(kowiki_dir, 'my_corpus.txt'), f"Data/nlp/my_vocab", vocab_size=1100)

print('최상위 file list : ', os.listdir("."))

# Sentencepe 확인
# load vocab
vocab = spm.SentencePieceProcessor()
vocab.load(os.path.join(data_dir, 'my_vocab.model'))

# vocab 출력
print(f"vocab's length : {len(vocab)}")
print('Id : Piece')
for id in range(16):
    print(f"{id:2d}: {vocab.id_to_piece(id)}")

# my_text를 tokenize 함
# sentence to pieces
pieces = vocab.encode_as_pieces(my_text)
print('pieces top 20 : \n', pieces[0:20])

# encode 적용 예시
#print(vocab.encode_as_pieces("나는 오늘 수원에 놀러 갔다."))
#print(vocab.encode_as_pieces("나는 오늘 배가많이고파서밥을많이먹은후 놀러 갔다."))

# tokenize된 값을 string 으로 복원
# pieces to sentence
# 결과 너무 길어서 주석처리, 확인 필요시 주석해제
# print(vocab.decode_pieces(pieces))

# tokenize된 값을 id로 변경
# piece to id
piece_ids = []
for piece in pieces:
    piece_ids.append(vocab.piece_to_id(piece))
print('piece -> id (top 50) : \n', piece_ids[0:50])

# my_text를 id로 tokenize 함
# piece로 나눈 후 id로 변환하는 2단계의 과정을 1단계로 축소
# sentence to ids
ids = vocab.encode_as_ids(my_text)
print(ids[0:50])

# tokenize된 id 값을 string 으로 복원
# id to sentence
# 결과 너무 길어서 주석처리, 확인 필요시 주석해제
# print(vocab.decode_ids(ids))

# id 값을 token으로 변경
# id to piece
id_pieces = []
for id in ids:
    id_pieces.append(vocab.id_to_piece(id))
print('id -> piece (top 50) : \n', id_pieces[0:50])