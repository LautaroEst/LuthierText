from .cooccurrence import word_by_word_cooccurrence
from .cooccurrence import word_by_document_cooccurrence
from .cooccurrence import word_by_category_cooccurrence
import numpy as np
from tqdm import tqdm
from ..reweighting import observed_over_expected, pmi, tfidf


def _append_unk(X,vocab,unk_token):

	if unk_token not in vocab.keys():
		X.resize(X.shape[0]+1,X.shape[1])
		unk_idx = X.shape[0]-1
	else:
		unk_idx = vocab[unk_token]
	
	return X, unk_idx

def _get_reweight_fn(reweighting):
	if reweighting == 'oe':
		return observed_over_expected
	elif reweighting == 'ppmi':
		return pmi
	elif reweighting == 'tfidf':
		return tfidf
	elif reweighting is None:
		return lambda x: x
	else:
		raise ValueError('{} reweighting method not implemented.'.format(reweighting))

class CBOWVectorizer(object):

	def __init__(self,func,**kwargs):
		self.func = func
		self.unk_token = kwargs.pop('unk_token')
		self.tokenizer = kwargs.pop('tokenizer')
		reweighting = kwargs.pop('reweighting',None)
		self.reweighting = _get_reweight_fn(reweighting)
		self.kwargs = kwargs

	def train(self,corpus,*args):
		X, vocab = self.func(corpus,*args,**self.kwargs)
		X, unk_idx = _append_unk(X,vocab,self.unk_token)
		X = self.reweighting(X.toarray())
		self.X = X
		self.vocab = vocab
		self.unk_idx = unk_idx

	def vectorize_doc(self,doc):
		vocab_get = self.vocab.get
		unk_idx = self.unk_idx
		X = self.X
		indices = [vocab_get(tk,unk_idx) for tk in self.tokenizer(doc)]
		cbow_vec = X[indices,:].sum(axis=0)
		return cbow_vec

	def vectorize_corpus(self,corpus):
		cbow_mat = np.zeros((len(corpus),self.X.shape[1]))
		for i,doc in enumerate(tqdm(corpus)):
			cbow_mat[i,:] = self.vectorize_doc(doc)
		return cbow_mat


class WordByWordVectorizer(CBOWVectorizer):

	def __init__(self,window=None, left_n=2, right_n=2, tokenizer=None, 
				min_count=0., max_count=None, max_words=None, unk_token=None,
				reweighting=None):

		kwargs = {
			'window': window,
			'left_n': left_n,
			'right_n': right_n,
			'tokenizer': tokenizer,
			'min_count': min_count,
			'max_count': max_count,
			'max_words': max_words,
			'unk_token': unk_token,
			'reweighting': reweighting
		}
		super().__init__(word_by_word_cooccurrence,**kwargs)

class WordByDocumentVectorizer(CBOWVectorizer):

	def __init__(self, tokenizer=None, min_count=0., max_count=None, 
				max_words=None, unk_token=None,reweighting=None):

		kwargs = {
			'tokenizer': tokenizer,
			'min_count': min_count,
			'max_count': max_count,
			'max_words': max_words,
			'unk_token': unk_token,
			'reweighting': reweighting
		}
		super().__init__(word_by_document_cooccurrence,**kwargs)


class WordByCategoryVectorizer(CBOWVectorizer):

	def __init__(self, tokenizer=None, min_count=0., max_count=None, 
				max_words=None, unk_token=None,reweighting=None):

		kwargs = {
			'tokenizer': tokenizer,
			'min_count': min_count,
			'max_count': max_count,
			'max_words': max_words,
			'unk_token': unk_token,
			'reweighting': reweighting
		}
		super().__init__(word_by_category_cooccurrence,**kwargs)
