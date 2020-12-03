from .cooccurrence import word_by_word_cooccurrence
from .cooccurrence import word_by_document_cooccurrence
from .cooccurrence import word_by_category_cooccurrence
import numpy as np
from tqdm import tqdm


def _append_unk(X,vocab,unk_token):

	if unk_token not in vocab.keys():
		X.resize(X.shape[0]+1,X.shape[1])
		unk_idx = X.shape[0]-1
	else:
		unk_idx = vocab[unk_token]
	
	return X, unk_idx

class CBOWVectorizer(object):

	def __init__(self,func,**kwargs):
		self.func = func
		self.unk_token = kwargs.pop('unk_token')
		self.tokenizer = kwargs.pop('tokenizer')
		self.kwargs = kwargs

	def train(self,corpus):
		X, vocab = self.func(corpus,**self.kwargs)
		X, unk_idx = _append_unk(X,vocab,self.unk_token)
		
		self.X = X
		self.vocab = vocab
		self.unk_idx = unk_idx

	def vectorize_doc(self,doc):
		vocab = self.vocab
		unk_idx = self.unk_idx
		X = self.X
		indices = [vocab.get(tk,unk_idx) for tk in self.tokenizer(doc)]
		cbow_vec = X[indices,:].sum(axis=0).A1
		return cbow_vec

	def vectorize_corpus(self,corpus):
		vocab = self.vocab
		unk_idx = self.unk_idx
		X = self.X
		cbow_mat = np.zeros((len(corpus),X.shape[1]))
		for i,doc in enumerate(tqdm(corpus)):
			indices = [vocab.get(tk,unk_idx) for tk in self.tokenizer(doc)]
			cbow_mat[i,:] = X[indices,:].sum(axis=0).A1
		return cbow_mat


class WordByWordVectorizer(CBOWVectorizer):

	def __init__(self,window=None, left_n=2, right_n=2, tokenizer=None, 
				min_count=0., max_count=None, max_words=None, unk_token=None):

		kwargs = {
			'window': window,
			'left_n': left_n,
			'right_n': right_n,
			'tokenizer': tokenizer,
			'min_count': min_count,
			'max_count': max_count,
			'max_words': max_words,
			'unk_token': unk_token
		}
		super().__init__(word_by_word_cooccurrence,**kwargs)

class WordByDocumentVectorizer(CBOWVectorizer):

	def __init__(self, tokenizer=None, min_count=0., max_count=None, 
				max_words=None, unk_token=None):

		kwargs = {
			'tokenizer': tokenizer,
			'min_count': min_count,
			'max_count': max_count,
			'max_words': max_words,
			'unk_token': unk_token
		}
		super().__init__(word_by_document_cooccurrence,**kwargs)


class WordByCategoryVectorizer(CBOWVectorizer):

	def __init__(self, tokenizer=None, min_count=0., max_count=None, 
				max_words=None, unk_token=None):

		kwargs = {
			'tokenizer': tokenizer,
			'min_count': min_count,
			'max_count': max_count,
			'max_words': max_words,
			'unk_token': unk_token
		}
		super().__init__(word_by_category_cooccurrence,**kwargs)

	def train(self,corpus,labels):
		X, vocab = self.func(corpus,labels,**self.kwargs)
		X, unk_idx = _append_unk(X,vocab,self.unk_token)

		self.X = X
		self.vocab = vocab
		self.unk_idx = unk_idx
