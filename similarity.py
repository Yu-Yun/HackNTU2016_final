# -*- coding: utf-8 -*-
import jieba
import os_weight
import train_svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from operator import itemgetter
### document into tuple format
#doc_file = open('data/allsents.txt','r').read()
#doc_split = doc_file.split('\n')
#doc_split = list(new_input) + doc_split
# TODO: check the variable name of the input text



def seg(sent):
    res = jieba.cut(sent)
    return ' '.join(res)


def corpus(new_input):
    doc_file = open('data/allsents.txt','r').read()
    doc_split = doc_file.split('\n')[:-1]
    doc_split = [new_input] + doc_split

    doc_seg = [seg(i) for i in doc_split]
    corpus_res = tuple(doc_seg)
    return corpus_res



def similarity(new_input):
    corpus_res = corpus(new_input)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_res)
    res = cosine_similarity(tfidf_matrix[0], tfidf_matrix).tolist()
    return sorted(res[0])[-2]


# TODO: 把 svm 的 func 引進來分析 unknown sent, 把預測出來的 label 餵入 sub_cat() 裡面找 key
# 少一個 new_label 的 variable name

def adjusted_sim(new_input):
    new_label = train_svm.sent_score(new_input)
    sub_cat = os_weight.sub_cat()
    os_res = sub_cat[new_label]
    adj_res = similarity(new_input)
    adjusted = []

    for i in os_res:
        final = float(i[1])*float(adj_res)
        adjusted.append([i[0],round(final,2)])

    return sorted(adjusted, key=itemgetter(1), reverse= True)
