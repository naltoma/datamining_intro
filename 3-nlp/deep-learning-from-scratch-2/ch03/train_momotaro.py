import collections, os, pickle, datetime
import nltk, MeCab
import numpy as np

# データの準備
filename = "./corpus/momotaro.txt"
with open(filename, "r") as fh:
    sentences = ""
    for line in fh.readlines():
        sentences += line + " "

def tokenize(sentences):
    """文章を分かち書きするとともに、ボキャブラリも返す。

    :param sentences(str): 複数の文章を含む文字列。日本語想定。
    :return(list):
      tokens(list): 分かち書きした単語をlistとしてまとめたもの。
      vocab(list): ボキャブラリ。ユニークな単語一覧。
    """

    # 「。」、「！」、「？」で終わる一連の文字列を文として認識し分割する。
    jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^　「」！？。]*[！？。]')
    tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    sents = jp_sent_tokenizer.tokenize(sentences)
    tokens = []
    vocab = []
    for sent in sents:
        node = tagger.parseToNode(sent)
        while node:
            features = node.feature.split(",")
            base_word = features[6]
            if base_word == "*" or base_word == " " or base_word == "\n":
                node = node.next
                continue
            tokens.append(base_word)
            if base_word not in vocab:
                vocab.append(base_word)
            node = node.next
    return tokens, vocab

# 文章からボキャブラリと分かち書きを用意。
filename = "./corpus/momotaro.txt"
with open(filename, "r") as fh:
    sentences = ""
    for line in fh.readlines():
        sentences += line + " "
wakati_sentences, vocab = tokenize(sentences)

print("vocab[:5]", vocab[:5])
print("len(vocab)", len(vocab))

# ボキャブラリと分かち書き文章から、データセットを作成。
word_to_id = dict((c,i) for i,c in enumerate(vocab))
id_to_word = dict((i,c) for i,c in enumerate(vocab))
print(char_to_id)

# 分かち書き文章を単語IDで表現
wakati_ids = []
for word in wakati_sentences:
    wakati_ids.append(word_to_id[word])

# データセットの構築
window_size = 5

def make_dataset(wakati_sents, char_to_id, window_size):
    length = len(char_to_id)
    X = [] # input = context words
    Y = [] # output = target word
    for index in range(len(wakati_sentences)):

        # 文脈ベクトルの準備
        begin = index - window_size
        end = index + window_size + 1
        temp = np.array([])
        for i in range(begin, index):
            if i >= 0:
                word_id = char_to_id[wakati_sents[i]]
                context = np.zeros(length)
                context[word_id] = 1
                temp = np.concatenate([temp,context], axis=0)
            else:
                context = np.zeros(length)
                temp = np.concatenate([temp,context], axis=0)
        for i in range(index+1, end):
            if i < length:
                word_id = char_to_id[wakati_sents[i]]
                context = np.zeros(length)
                context[word_id] = 1
                temp = np.concatenate([temp, context], axis=0)
            else:
                context = np.zeros(length)
                temp = np.concatenate([temp, context], axis=0)
        X.append(temp[:])

        # 対象単語ベクトルの準備
        target_id = char_to_id[wakati_sents[index]]
        temp = np.zeros(length)
        temp[target_id] = 1
        Y.append(temp[:])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

X, Y = make_dataset(wakati_sentences, char_to_id, window_size)
print("len(X)={}, len(Y)={}".format(len(X), len(Y)))
print("X.shape={}, Y.shape={}".format(X.shape, Y.shape))
print("len(X[0])=", len(X[0]))






# coding: utf-8
import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot


#window_size = 1
hidden_size = 5
batch_size = 10
max_epoch = 200

#text = 'You say goodbye and I say hello.'
#corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(vocab)
#contexts, target = create_contexts_target(corpus, window_size)
#target = convert_one_hot(target, vocab_size)
#contexts = convert_one_hot(contexts, vocab_size)
contexts, target = create_contexts_target(wakati_ids, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
#trainer.fit(X, Y, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
