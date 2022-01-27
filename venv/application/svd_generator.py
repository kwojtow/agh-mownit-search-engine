import pickle
import nltk
import numpy as np
import bs4 as bs
import urllib.request
import re
import heapq
from scipy import sparse
import sys
import gc

sys.setrecursionlimit(10000)


def csr_vappend(a, b):
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a


dictionary = pickle.load(open("dictionary.p", "rb"))
articles_vectors = pickle.load(open("standard.p", "rb"))

N = articles_vectors.shape[1]


def idf(w, nw):
    if nw != 0:
        return np.log10(N / nw)
    return 1


for i in range(N):
    nw = articles_vectors[i].count_nonzero()
    articles_vectors[i] *= idf(i, nw)


def normalize():
    global articles_vectors
    global query_vec
    for i in range(N):
        articles_vectors[:, i] = articles_vectors[:, i] / sparse.linalg.norm(articles_vectors[:, i])


normalize()

k_val = 30
U, s, Vt = sparse.linalg.svds(articles_vectors, k=k_val)
reduced_articles_vectors = np.zeros((articles_vectors.shape[0], articles_vectors.shape[1]))
for i in range(k_val):
    reduced_articles_vectors = reduced_articles_vectors + s[i] * np.outer(np.transpose(U)[i], Vt[i])
pickle.dump(sparse.csr_matrix(reduced_articles_vectors), open("reduced4.p", "wb"))
