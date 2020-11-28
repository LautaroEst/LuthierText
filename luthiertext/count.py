from collections import Counter, defaultdict
from itertools import tee, islice
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np


def get_ngrams(doc, ngram_range=(1,1)):

	for n in range(ngram_range[0],ngram_range[1]+1):
	    tlst = doc
	    while True:
	        a, b = tee(tlst)
	        l = tuple(islice(a, n))
	        if len(l) == n:
	            yield ' '.join(l)
	            next(b)
	            tlst = b
	        else:
	            break


def count_bag_of_ngrams(corpus, ngram_range=(1,1), tokenizer=None):
	
	if tokenizer is None:
		tokenizer = lambda x: x

	data = []
	indices = []
	indptr = [0]

	full_vocab = defaultdict()
	full_vocab.default_factory = full_vocab.__len__

	for doc in tqdm(corpus):
		features = dict(Counter(get_ngrams(tokenizer(doc),ngram_range)))
		data.extend(features.values())
		indices.extend([full_vocab[tk] for tk in features])
		indptr.append(len(indices))

	vocab_len = len(full_vocab)
	X = csr_matrix((data,indices,indptr),shape=(len(corpus),vocab_len))
	return X, dict(full_vocab)



def reduce_by_freq(X, vocab, min_count=0., max_count=None, max_words=None):
	"""
	Recibe la matriz de coocurrencias X y el vocabulario de esa matriz. El diccionario
	vocab en general viene de count_bag_of_ngrams y por lo tanto se encuentra ordenado 
	por aparición de palabras en el corpus, por lo cual no hace falta ordenarlo. La 
	función filtra las pimeras max_words palabras que tienen una frecuencia entre 
	min_count y max_count.
	"""

	# Obtengo los tokens ordenados según la matriz de ocurrencias
	keys = np.array(list(vocab.keys()))
	values = np.array(list(vocab.values()))
	keys = keys[np.argsort(values)]

	# Ordeno por frecuencia
	freqs = np.array(X.sum(axis=1)).reshape(-1)
	sorted_idx = np.argsort(freqs)[::-1]
	
	sorted_freqs = freqs[sorted_idx]
	X = X[sorted_idx,:]
	keys = keys[sorted_idx]
	
	if min_count <= 0 and max_count is None:
		pass

	else:
		if max_count is None:
			max_count = np.inf

		mask = np.logical_and(sorted_freqs <= max_count, sorted_freqs >= min_count)
		X = X[mask,:]
		keys = keys[mask]

	if max_words is not None:
		X = X[:max_words,:]
		keys = keys[:max_words]

	vocab = {tk:idx for idx,tk in enumerate(keys)}
	return X, vocab