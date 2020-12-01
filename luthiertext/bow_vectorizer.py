import numpy as np
from collections import Counter, defaultdict
from itertools import chain, tee, islice
from scipy.sparse import csr_matrix
from tqdm import tqdm


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


def count_bag_of_ngrams_from_stratch(corpus, ngram_range, tokenizer, unk_token):
	
	# Definiciones para la matriz de coocurrencias:
	data = []
	indices = []
	indptr = [0]

	# Definiciones para el vocabulario:
	full_vocab = defaultdict()
	full_vocab.default_factory = full_vocab.__len__

	# Cuento palabras:
	for doc in tqdm(corpus):
		features = dict(Counter(get_ngrams(tokenizer(doc),ngram_range)))
		data.extend(features.values())
		indices.extend([full_vocab[tk] for tk in features])
		indptr.append(len(indices))

	# Armo la matriz y el diccionario de vocabulario:
	full_vocab = dict(full_vocab)
	if isinstance(unk_token,str):
		full_vocab[unk_token] = len(full_vocab)
	vocab_len = len(full_vocab)
	X = csr_matrix((data,indices,indptr),shape=(len(corpus),vocab_len))
	return X, full_vocab


def count_bag_of_ngrams_using_vocab(corpus, vocab, ngram_range=(1,1), 
						tokenizer=None, unk_token=None):
	
	# Definiciones para la matriz de coocurrencias:
	data = []
	indices = []
	indptr = [0]

	# Cuento palabras:
	unk_idx = len(vocab)
	for doc in tqdm(corpus):
		features = dict(Counter(get_ngrams(tokenizer(doc),ngram_range)))
		data.extend(features.values())
		indices.extend([vocab.get(tk,unk_idx) for tk in features])
		indptr.append(len(indices))

	# Armo la matriz y el diccionario de vocabulario:
	X = csr_matrix((data,indices,indptr),shape=(len(corpus),len(vocab)+1))

	if isinstance(unk_token,str):
		vocab[unk_token] = unk_idx	
	else:
		X = X[:,:-1]
	
	return X, vocab


def filter_by_counts(X,full_vocab,min_count,max_count,max_words,unk_token):

	freqs = X.sum(axis=0).A1
	sorted_indices = np.argsort(freqs)[::-1]
	sorted_frequencies = freqs[sorted_indices]

	if min_count <= 0 and max_count is None:
		mask = np.ones(X.shape[1],dtype=bool)

	else:
		if max_count is None:
			max_count = np.inf
		mask = np.logical_and(sorted_frequencies <= max_count, 
								sorted_frequencies >= min_count)

	sorted_indices = sorted_indices[mask]

	if max_words is not None:
		sorted_indices = sorted_indices[:max_words]

	if isinstance(unk_token,str):
		unk_idx = full_vocab[unk_token]
		if unk_idx not in sorted_indices:
			if max_words is not None:
				if len(sorted_indices) < max_words:
					sorted_indices = np.hstack((sorted_indices,np.array([unk_idx])))
				else:
					sorted_indices[-1] = unk_idx
			else:
				sorted_indices = np.hstack((sorted_indices,np.array([unk_idx])))
		
		mask = np.ones(X.shape[1],dtype=bool)
		mask[sorted_indices] = False
		X[:,unk_idx] = X[:,mask].sum(axis=1)

	X = X[:,sorted_indices]
	idx_to_tk = {idx:tk for tk,idx in full_vocab.items()}
	vocab = {idx_to_tk[idx]:i for i,idx in enumerate(sorted_indices)}

	return X, vocab




class BONgramsVectorizer(object):
	""" Vectorizer para convertir texto en vectores.
	
	Tiene dos formas de uso: cuando vocab=None, recibe min_count, max_count, 
	max_words y ngram_range para obtener un vocabulario a partir del corpus
	con todas esas limitaciones. Cuando vocab es una lista de palabras, cuenta 
	sólo esas palabras, y si unk_token es un string, entonces lo incluye en el 
	vocabulario. """

	def __init__(self, tokenizer=None, min_count=0., 
				max_count=None, max_words=None, ngram_range=(1,1),
				vocab=None, unk_token=None):

		if tokenizer is None:
			self.tokenizer = lambda x: x
		elif not callable(tokenizer):
			raise ValueError('Tokenizer must be callable or None.')
		else:
			self.tokenizer = tokenizer
		
		self.min_count = min_count
		self.max_count = max_count
		self.max_words = max_words
		self.ngram_range = ngram_range

		if vocab is None:
			self.infer_vocab = True
		else:
			self.vocab = {tk:idx for idx,tk in enumerate(vocab)}
			self.infer_vocab = False

		if unk_token is None or isinstance(unk_token,str):
			self.unk_token = unk_token
		else:
			raise ValueError('unk_token must be string or None')


	def fit_transform(self,corpus):
		
		if self.infer_vocab:
			X, full_vocab = count_bag_of_ngrams_from_stratch(corpus, 
										self.ngram_range, 
										self.tokenizer,
										self.unk_token)
		
		else:
			X, full_vocab = count_bag_of_ngrams_using_vocab(corpus,
										self.vocab, 
										self.ngram_range, 
										self.tokenizer,
										self.unk_token)
		
		X, vocab = filter_by_counts(X,
									full_vocab,
									self.min_count,
									self.max_count,
									self.max_words,
									self.unk_token)
		self.vocab = vocab
		
		return X


	def fit(self,corpus):
		_ = self.fit_transform(corpus)

	
	def transform(self,corpus):

		vocab = self.vocab
		unk_token = self.unk_token
		tokenizer = self.tokenizer
		ngram_range = self.ngram_range

		# Definiciones para la matriz de coocurrencias:
		data = []
		indices = []
		indptr = [0]

		# Cuento palabras:
		unk_idx = len(vocab)
		for doc in tqdm(corpus):
			features = dict(Counter(get_ngrams(tokenizer(doc),ngram_range)))
			data.extend(features.values())
			indices.extend([vocab.get(tk,unk_idx) for tk in features])
			indptr.append(len(indices))

		# Armo la matriz y el diccionario de vocabulario:
		X = csr_matrix((data,indices,indptr),shape=(len(corpus),len(vocab)+1)).tolil()

		if isinstance(unk_token,str):
			X[:,vocab[unk_token]] = X[:,-1]
			X.resize(X.shape[0],X.shape[1]-1)
		else:
			X = X[:,:-1]

		return X.tocsr()
		


