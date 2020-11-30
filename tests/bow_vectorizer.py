import sys
sys.path.append('../')
from luthiertext import BONgramsVectorizer

import re
from itertools import chain, tee, islice
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

if __name__ == '__main__':
	test1()
	test2()