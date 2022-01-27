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

dictionary = []
urls = []
titles = []
wordfreq = {}
articles_vectors = None
corpus_max_size = 1000


def csr_vappend(a, b):
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a


def crawl(url):
    if len(urls) >= corpus_max_size:
        return
    if url in urls:
        return
    urls.append(url)
    raw_html = urllib.request.urlopen(url)
    raw_html = raw_html.read()
    article_html = bs.BeautifulSoup(raw_html, 'html.parser')
    article_links = article_html.find_all('a', attrs={'href': re.compile("(?!^/wiki/.*:.*)(^/wiki/)")})
    title = article_html.find('title').text
    titles.append(title)
    print("Crawling ", len(urls), ": ", title)
    for link in article_links:
        crawl('https://en.wikipedia.org' + link.get('href'))


def scrap(url):
    raw_html = urllib.request.urlopen(url)
    raw_html = raw_html.read()

    article_html = bs.BeautifulSoup(raw_html, 'html.parser')

    article_content = article_html.find_all(['p', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7'])

    article_text = ''

    for seg in article_content:
        article_text += seg.text

    article_text = re.sub(r'\W', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)
    article_tokens = nltk.word_tokenize(article_text)
    ps = nltk.stem.PorterStemmer()
    article_tokens = [token for token in article_tokens if not token in nltk.corpus.stopwords.words('english')]
    article_tokens = [ps.stem(token, to_lowercase=True) for token in article_tokens]

    return article_tokens


def prepare_dictionary():
    global dictionary
    wordfreq = {}
    iter = 0
    for url in urls:
        iter += 1
        print("Preparing dictionary ", iter, ": ", url)
        tokens = scrap(url)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    dictionary = heapq.nlargest(len(wordfreq) // 2, wordfreq, key=wordfreq.get)


def prepare_bag_of_words():
    global articles_vectors
    iter = 0
    for url in urls:
        iter += 1
        print("Preparing bag-of-words ", iter, ": ", url)
        article_tokens = scrap(url)
        art_vec = []
        for token in dictionary:
            art_vec.append(article_tokens.count(token))

        if articles_vectors is None:
            articles_vectors = sparse.csr_matrix(art_vec, dtype=np.float64)
        else:
            articles_vectors = csr_vappend(articles_vectors, sparse.csr_matrix(art_vec))

    articles_vectors = articles_vectors.transpose()


crawl('https://en.wikipedia.org/wiki/Main_Page')
pickle.dump(urls, open("urls.p", "wb"))
pickle.dump(titles, open("titles.p", "wb"))
del titles
gc.collect()

prepare_dictionary()
pickle.dump(dictionary, open("dictionary.p", "wb"))

prepare_bag_of_words()
pickle.dump(articles_vectors, open("standard.p", "wb"))

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
pickle.dump(articles_vectors, open("normalized.p", "wb"))

k_val = 30
U, s, Vt = sparse.linalg.svds(articles_vectors, k=k_val)
reduced_articles_vectors = np.zeros((articles_vectors.shape[0], articles_vectors.shape[1]))
for i in range(k_val):
    reduced_articles_vectors = reduced_articles_vectors + s[i] * np.outer(np.transpose(U)[i], Vt[i])
pickle.dump(sparse.csr_matrix(reduced_articles_vectors), open("reduced.p", "wb"))
