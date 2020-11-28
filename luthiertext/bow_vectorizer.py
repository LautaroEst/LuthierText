from .count import count_bag_of_ngrams, reduce_by_freq, get_ngrams
import numpy as np
from collections import Counter
from itertools import chain
from scipy.sparse import csr_matrix


def count_from_existing_vocab(X,full_vocab,my_vocab):
	existing_values = []
	old_tokens = []
	new_tokens = []
	for tk in my_vocab:
		try: 
			existing_values.append(full_vocab[tk])
			old_tokens.append(tk)
		except KeyError:
			new_tokens.append(tk)
		
	X = X[existing_values,:].tocsr()
	X.resize(X.shape[0]+len(new_tokens),X.shape[1])
	my_vocab = {tk:idx for idx,tk in enumerate(chain(old_tokens,new_tokens))}
	X, my_vocab = reduce_by_freq(X, my_vocab, 0, None, None)
	return X, my_vocab


class BONgramsVectorizer(object):
	""" Vectorizer para convertir texto en vectores.
	
	Tiene dos formas de uso: cuando vocab=None, recibe min_count, max_count, 
	max_words y ngram_range para obtener un vocabulario a partir del corpus
	con todas esas limitaciones. Cuando vocab es una lista de palabras, cuenta 
	sólo esas palabras. No tiene implementado aun el tema del token UNK. """

	def __init__(self,tokenizer=None, min_count=0., 
				max_count=None, max_words=None,ngram_range=(1,1),vocab=None):
		self.tokenizer = tokenizer
		self.min_count = min_count
		self.max_count = max_count
		self.max_words = max_words
		self.ngram_range = ngram_range
		self.vocab = vocab

	def fit_transform(self,corpus):
		X, full_vocab = count_bag_of_ngrams(corpus,self.ngram_range,self.tokenizer)

		if self.vocab is None:
			X, vocab = reduce_by_freq(X.T.tocsr(), full_vocab, self.min_count, self.max_count, self.max_words)
			self.vocab = vocab	
		else:
			X, vocab = count_from_existing_vocab(X.T.tocsr(),full_vocab,self.vocab)
			self.vocab = vocab
			
		return X.T.tocsr()

	def fit(self,corpus):
		_ = self.fit_transform(corpus)
	
	def transform(self,corpus):
		
		if self.tokenizer is None:
			tokenizer = lambda x: x
		else:
			tokenizer = self.tokenizer

		ngram_range = self.ngram_range
		vocab = self.vocab

		data = []
		indices = []
		indptr = [0]

		for doc in corpus:
			features = dict(Counter(get_ngrams(tokenizer(doc),ngram_range)))
			data_extend_list = []
			indices_extend_list = []
			for f in features:
				if f in vocab:
					data_extend_list.append(features[f])
					indices_extend_list.append(vocab[f])
			data.extend(data_extend_list)
			indices.extend(indices_extend_list)
			indptr.append(len(indices))

		X = csr_matrix((data,indices,indptr),shape=(len(corpus),len(vocab)))
		return X







