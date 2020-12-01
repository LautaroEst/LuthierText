import sys
sys.path.append('../')
from luthiertext import BONgramsVectorizer

import re
from itertools import chain, tee, islice
from collections import Counter
import numpy as np


sample_train_corpus = ['La zamba "Añoralgias" ha sido recopilada por un gran investigador de nuestro folklore.',
	'Un hombre nacido en el norte, el noruego Sven Kundsen, el payo Kundsen.',
	'Solía decir: "Yo soy más criollo que el bacalau".',
	'Cuando le pedían su opinión sobre algún tema comprometido, él respondía: "Yo argentino".',
	'Arqueólogo, musicólogo, viajero "infatigólogo", a su iniciativa debemos el simposio interdisciplinario que reunió a folkloristas y ginecólogos.',
	'El tema era "La relación entre el examen de mama y el alazán de tata".',
	'Fue en un pueblito de Salta donde Kundsen oyó por primera vez la zamba "Añoralgias" que escucharemos, cantada por una aciana de ciento ocho años a la que había encontrado en una de sus excavaciones arqueológicas.',
	'Me dijo: "Mire ese algarrobo", señalando un guanaco.',
	'En efecto, se confundía con el paisaje.',
	'Cuando ella terminó de canturrear la zamba, sigue diciendo Kundsen, le pregunté si la había aprendido de sus abuelos.',
	'Y ella me contestó: "Esta zamba la aprendí en un compact que me mandaron de Buenos Aires".',
	]

sample_test_corpus = ['A pesar de su origen escandinavo, Kundsen amaba a nuestra tierra.',
	'Dice Kundsen en sus memorias: La venerable mujer parecía confundirse con el paisaje.',	
	'A continuación escucharemos, en versión de Les Luthiers, la zamba "Añoralgias".'
	]

dicts_test = [{'A': 1, 'pesar': 1, 'de': 1, 'su': 1, 'origen': 1, 'escandinavo': 1, 
			',': 1, 'Kundsen': 1, 'amaba': 1, 'a': 1, 'nuestra': 1, 'tierra': 1, '.': 1},
			{'Dice': 1, 'Kundsen': 1, 'en': 1, 'sus': 1, 'memorias': 1, ':': 1, 'La': 1, 
			'venerable': 1, 'mujer': 1, 'parecía': 1, 'confundirse': 1, 'con': 1, 'el': 1, 
			'paisaje': 1, '.':1},
			{'A': 1, 'continuación': 1, 'escucharemos': 1, ',': 2, 'en': 1, 'versión': 1, 'de': 1,
			'Les': 1, 'Luthiers': 1, 'la': 1, 'zamba': 1, '"': 2, 'Añoralgias': 1, '.': 1}]

pattern = re.compile(r'[a-zA-ZñÑáéíóúÁÉÍÓÚüÜ]+|[0-9]+|[":,]|[\.]+')
tokenizer = lambda x: pattern.findall(x)

def test1():
	splitted_text = [tokenizer(text) for text in sample_train_corpus]
	words_vocab = sorted(list(set(chain.from_iterable(splitted_text))))
	
	vec = BONgramsVectorizer(tokenizer=tokenizer,
							 min_count=0.,
							 max_count=None,
							 ngram_range=(1,1),
							 vocab=None,
							 unk_token=None)

	vec.fit(sample_train_corpus)
	X = vec.transform(sample_test_corpus)

	try:
		vocab = vec.vocab
		assert (sorted(list(vocab.keys())) == words_vocab)
		for i,doc in enumerate(dicts_test):
			for feature in doc:
				try:
					assert X[i,vocab[feature]] == doc[feature]
				except KeyError:
					pass
		print('Test 1 passed!')
	except AssertionError:
		print('Test 1 failed: las palabras no son las mismas')


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

def test2():
	splitted_text = [tokenizer(text) for text in sample_train_corpus]
	words_vocab = sorted(list(set(chain.from_iterable(splitted_text))) + ['UNK']) 
	
	vec = BONgramsVectorizer(tokenizer=tokenizer,
							 min_count=0.,
							 max_count=None,
							 max_words=None,
							 ngram_range=(1,1),
							 vocab=None,
							 unk_token='UNK')

	vec.fit(sample_train_corpus)
	X = vec.transform(sample_test_corpus)

	try:
		vocab = vec.vocab
		assert (sorted(list(vocab.keys())) == words_vocab)
		for i,doc in enumerate(dicts_test):
			unks = 0
			for feature in doc:
				try:
					assert X[i,vocab[feature]] == doc[feature]
				except KeyError:
					unks += doc[feature]
			assert unks == X[i,vocab['UNK']]
		print('Test 2 passed!')
	except AssertionError:
		print('Test 2 failed')


def test3():

	ngram_range = (1,2)
	
	vec = BONgramsVectorizer(tokenizer=tokenizer,
							 min_count=0.,
							 max_count=None,
							 max_words=None,
							 ngram_range=ngram_range,
							 vocab=None,
							 unk_token='UNK')

	X_train = vec.fit_transform(sample_train_corpus)
	X_test = vec.transform(sample_test_corpus)

	train_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_train_corpus]
	true_vocab = []
	for features in train_features:
		true_vocab.extend(list(features.keys()))
	test_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_test_corpus]
	new_test_features = []
	for features in test_features:
		feat_dict = {}
		for feature in features.keys():
			if feature in true_vocab:
				feat_dict[feature] = features[feature]
			else:
				try:
					feat_dict['UNK'] += features[feature]
				except:
					feat_dict['UNK'] = features[feature]
		new_test_features.append(feat_dict)

	try:
		for i,features in enumerate(train_features):
			for feature in features.keys():
				assert X_train[i,vec.vocab[feature]] == features[feature]
		for i,features in enumerate(new_test_features):
			for feature in features.keys():
				assert X_test[i,vec.vocab[feature]] == features[feature]
		
		print('Test 3 passed!')
	except AssertionError:
		print('Test 3 failed')


def test4():

	ngram_range = (1,2)
	
	vec = BONgramsVectorizer(tokenizer=tokenizer,
							 min_count=0.,
							 max_count=3,
							 max_words=None,
							 ngram_range=ngram_range,
							 vocab=None,
							 unk_token=None)

	X_train = vec.fit_transform(sample_train_corpus)
	X_test = vec.transform(sample_test_corpus)

	train_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_train_corpus]
	freqs_dict = {}
	for features in train_features:
		for f in features.keys():
			try:
				freqs_dict[f] += features[f]
			except:
				freqs_dict[f] = features[f]

	true_vocab = [f for f in freqs_dict.keys() if freqs_dict[f] >=0 and freqs_dict[f] <=3]

	new_train_features = []
	for features in train_features:
		feat_dict = {}
		for feature in features.keys():
			if feature in true_vocab:
				feat_dict[feature] = features[feature]
		new_train_features.append(feat_dict)

	test_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_test_corpus]
	new_test_features = []
	for features in test_features:
		feat_dict = {}
		for feature in features.keys():
			if feature in true_vocab:
				feat_dict[feature] = features[feature]
		new_test_features.append(feat_dict)

	try:
		for i,features in enumerate(new_train_features):
			for feature in features.keys():
				assert X_train[i,vec.vocab[feature]] == features[feature]
		for i,features in enumerate(new_test_features):
			for feature in features.keys():
				assert X_test[i,vec.vocab[feature]] == features[feature]
		
		print('Test 4 passed!')
	except AssertionError:
		print('Test 4 failed')


def test5():

	ngram_range = (1,1)
	max_words = 5
	min_count = 0
	max_count = 3
	
	vec = BONgramsVectorizer(tokenizer=tokenizer,
							 min_count=min_count,
							 max_count=max_count,
							 max_words=max_words,
							 ngram_range=ngram_range,
							 vocab=None,
							 unk_token='UNK')

	X_train = vec.fit_transform(sample_train_corpus)
	X_test = vec.transform(sample_test_corpus)

	train_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_train_corpus]
	freqs_dict = {}
	for features in train_features:
		for f in features.keys():
			try:
				freqs_dict[f] += features[f]
			except:
				freqs_dict[f] = features[f]

	""" keys = np.array(list(freqs_dict.keys()))
	freqs = np.array(list(freqs_dict.values()))
	ids = np.argsort(freqs)[::-1]
	freqs_dict = {tk:idx for idx,tk in zip(freqs[ids],keys[ids])}
	true_vocab = [f for f in freqs_dict.keys() if freqs_dict[f] >=min_count and freqs_dict[f] <=max_count][:max_words] """
	true_vocab = ['por', 'a', 'su', 'una', 'UNK']

	new_train_features = []
	for features in train_features:
		feat_dict = {}
		for feature in features.keys():
			if feature in true_vocab:
				feat_dict[feature] = features[feature]
			else:
				try:
					feat_dict['UNK'] += features[feature]
				except:
					feat_dict['UNK'] = features[feature]
		new_train_features.append(feat_dict)

	test_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_test_corpus]
	new_test_features = []
	for features in test_features:
		feat_dict = {}
		for feature in features.keys():
			if feature in true_vocab:
				feat_dict[feature] = features[feature]
			else:
				try:
					feat_dict['UNK'] += features[feature]
				except:
					feat_dict['UNK'] = features[feature]
		new_test_features.append(feat_dict)

	try:
		for i,features in enumerate(new_train_features):
			for feature in features.keys():
				assert X_train[i,vec.vocab[feature]] == features[feature]
		for i,features in enumerate(new_test_features):
			for feature in features.keys():
				assert X_test[i,vec.vocab[feature]] == features[feature]
		
		print('Test 5 passed!')
	except AssertionError:
		print('Test 5 failed')


def test6():

	ngram_range = (1,1)
	max_words = None
	min_count = 0
	max_count = None
	vocab = ['por', 'a', 'su', 'una']
	
	vec = BONgramsVectorizer(tokenizer=tokenizer,
							 min_count=min_count,
							 max_count=max_count,
							 max_words=max_words,
							 ngram_range=ngram_range,
							 vocab=vocab,
							 unk_token='UNK')

	X_train = vec.fit_transform(sample_train_corpus)
	X_test = vec.transform(sample_test_corpus)

	train_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_train_corpus]
	freqs_dict = {}
	for features in train_features:
		for f in features.keys():
			try:
				freqs_dict[f] += features[f]
			except:
				freqs_dict[f] = features[f]

	true_vocab = vocab + ['UNK']

	new_train_features = []
	for features in train_features:
		feat_dict = {}
		for feature in features.keys():
			if feature in true_vocab:
				feat_dict[feature] = features[feature]
			else:
				try:
					feat_dict['UNK'] += features[feature]
				except:
					feat_dict['UNK'] = features[feature]
		new_train_features.append(feat_dict)

	test_features = [dict(Counter(get_ngrams(tokenizer(doc),ngram_range))) for doc in sample_test_corpus]
	new_test_features = []
	for features in test_features:
		feat_dict = {}
		for feature in features.keys():
			if feature in true_vocab:
				feat_dict[feature] = features[feature]
			else:
				try:
					feat_dict['UNK'] += features[feature]
				except:
					feat_dict['UNK'] = features[feature]
		new_test_features.append(feat_dict)

	try:
		for i,features in enumerate(new_train_features):
			for feature in features.keys():
				assert X_train[i,vec.vocab[feature]] == features[feature]
		for i,features in enumerate(new_test_features):
			for feature in features.keys():
				assert X_test[i,vec.vocab[feature]] == features[feature]
		
		print('Test 6 passed!')
	except AssertionError:
		print('Test 6 failed')


if __name__ == '__main__':
	""" test1()
	test2()
	test3()
	test4()
	test5() """
	test6()
