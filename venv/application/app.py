import eel
import pickle
import nltk
import numpy as np
import bs4 as bs
import urllib.request
import re
import heapq
from scipy import sparse

final_results = []
urls = pickle.load(open("urls.p", "rb"))
titles = pickle.load(open("titles.p", "rb"))
dictionary = pickle.load(open("dictionary.p", "rb"))


def find(query):
    global final_results
    global urls
    global dictionary
    final_results = []
    qtokens = nltk.word_tokenize(query)
    ps = nltk.PorterStemmer()
    qtokens = [ps.stem(token, to_lowercase=True) for token in qtokens]

    query_vec = []
    for token in dictionary:
        query_vec.append(qtokens.count(token))

    query_vec = sparse.csr_matrix(query_vec, dtype=np.float64)

    articles_vectors = pickle.load(open("standard.p", "rb"))

    results = {}
    for i in range(articles_vectors.shape[1]):
        divider = (sparse.linalg.norm(query_vec) * sparse.linalg.norm(articles_vectors[:, i]))
        if divider != 0:
            results[i] = ((query_vec @ articles_vectors[:, i]) / divider).toarray()[0][0]
        else:
            results[i] = 0

    best_results = heapq.nlargest(10, results, key=results.get)

    normal_results = []
    for r in best_results:
        res = []
        res.append(urls[r])
        res.append(titles[r])
        res.append(results[r])
        normal_results.append(res)

    print("Normal results finished")

    articles_vectors = pickle.load(open("normalized.p", "rb"))

    results = {}
    res = query_vec @ articles_vectors
    for i in range(res.shape[1]):
        results[i] = res[0, i]

    best_results = heapq.nlargest(10, results, key=results.get)

    normalized_results = []
    for r in best_results:
        res = []
        res.append(urls[r])
        res.append(titles[r])
        res.append(results[r])
        normalized_results.append(res)

    print("Normalized results finished")

    # reduced_articles_vectors = pickle.load(open("reduced.p", "rb"))
    #
    # results = {}
    # for i in range(reduced_articles_vectors.shape[1]):
    #     divider = (sparse.linalg.norm(query_vec) * sparse.linalg.norm(reduced_articles_vectors[:, i]))
    #     if divider != 0:
    #         results[i] = ((query_vec @ reduced_articles_vectors[:, i]) / divider).toarray()[0][0]
    #     else:
    #         results[i] = 0
    #
    # best_results = heapq.nlargest(10, results, key=results.get)

    reduced_results = []
    for r in best_results:
        res = []
        res.append(urls[r])
        res.append(titles[r])
        res.append(results[r])
        reduced_results.append(res)

    print("Reduced results finished")

    final_results.append(normal_results)
    final_results.append(normalized_results)
    final_results.append(reduced_results)


eel.init('web')


@eel.expose
def get_results(x):
    find(x)
    return final_results


eel.start('index.html',
          host='localhost',
          port=27000,
          cmdline_args=['--browser-startup-dialog'])