import sys
sys.path.append('../')

import re
from itertools import chain
import numpy as np

from luthiertext.word_vectors import word_by_document_cooccurrence as wdc




sample_corpus = ['La zamba "Añoralgias" ha sido recopilada por un gran investigador de nuestro folklore.',
 'Un hombre nacido en el norte, el noruego Sven Kundsen, el payo Kundsen.',
 'A pesar de su origen escandinavo, Kundsen amaba a nuestra tierra.',
 'Solía decir: "Yo soy más criollo que el bacalau".',
 'Cuando le pedían su opinión sobre algún tema comprometido, él respondía: "Yo argentino".',
 'Arqueólogo, musicólogo, viajero "infatigólogo", a su iniciativa debemos el simposio interdisciplinario que reunió a folkloristas y ginecólogos.',
 'El tema era "La relación entre el examen de mama y el alazán de tata".',
 'Fue en un pueblito de Salta donde Kundsen oyó por primera vez la zamba "Añoralgias" que escucharemos, cantada por una aciana de ciento ocho años a la que había encontrado en una de sus excavaciones arqueológicas.',
 'Dice Kundsen en sus memorias: La venerable mujer parecía confundirse con el paisaje.',
 'Me dijo: "Mire ese algarrobo", señalando un guanaco.',
 'En efecto, se confundía con el paisaje.',
 'Cuando ella terminó de canturrear la zamba, sigue diciendo Kundsen, le pregunté si la había aprendido de sus abuelos.',
 'Y ella me contestó: "Esta zamba la aprendí en un compact que me mandaron de Buenos Aires".',
 'A continuación escucharemos, en versión de Les Luthiers, la zamba "Añoralgias".']

pattern = re.compile(r'[a-zA-ZñÑáéíóúÁÉÍÓÚüÜ]+|[0-9]+|[":,]|[\.]+')
tokenizer = lambda x: pattern.findall(x)

def test1():
	splitted_text = [tokenizer(text) for text in sample_corpus]
	words_vocab = sorted(list(set(chain.from_iterable(splitted_text))))
	X, vocab = wdc(sample_corpus, tokenizer=tokenizer, 
	min_count=0., max_count=None, max_words=None)

	try:
		assert (sorted(list(set(vocab.keys()))) == words_vocab)
		print('Test 1 passed!')
	except AssertionError:
		print('Test 1 failed: las palabras no son las mismas')


def test2():
	splitted_text = [tokenizer(text) for text in sample_corpus]
	X, vocab = wdc(splitted_text,
					tokenizer=None,
					min_count=0,
					max_count=None,
					max_words=None)
	
	zamba_vec = np.zeros(len(sample_corpus))
	for idx in [0, 7, 11, 12, 13]:
		zamba_vec[idx] = 1
	
	try:
		assert np.array_equal(X[vocab['zamba'],:].toarray().reshape(-1), zamba_vec)
		print('Test 2 passed!')
	except AssertionError:
		print('Test 2 failed: las frecuencias de "zamba" no son las mismas')
	 

def test3():
	splitted_text = [tokenizer(text) for text in sample_corpus]
	X, vocab = wdc(splitted_text,
					tokenizer=None,
					min_count=0,
					max_count=None,
					max_words=None)
	freq_dict = {tk:0 for tk in vocab.keys()}
	for tk in chain.from_iterable(splitted_text):
		freq_dict[tk] += 1
	
	try:
		assert np.array_equal(X.sum(axis=1).A1.reshape(-1),np.array(list(freq_dict.values())))
		print('Test 3 passed!')
	except AssertionError:
		print('Test 3 failed: las frecuencias no están bien contadas')
	
def test4():
	splitted_text = [tokenizer(text) for text in sample_corpus]
	X, vocab = wdc(splitted_text,
					tokenizer=None,
					min_count=1,
					max_count=2,
					max_words=None)
	freq_dict = {tk:0 for tk in set(chain.from_iterable(splitted_text))}
	for tk in chain.from_iterable(splitted_text):
		freq_dict[tk] += 1
	one_or_two_freq = []
	for tk,freq in freq_dict.items():
		if freq in (1,2):
			one_or_two_freq.append(tk)
	
	try:
		assert sorted(one_or_two_freq) == sorted(list(vocab.keys()))
		print('Test 4 passed!')
	except AssertionError:
		print('Test 4 failed: las frecuencias no están bien contadas')

def test5():
	splitted_text = [tokenizer(text) for text in sample_corpus]
	X, vocab = wdc(splitted_text,
					tokenizer=None,
					min_count=1,
					max_count=2,
					max_words=5)
	try:
		for tk in vocab.keys():
			assert X.sum(axis=1).A1.reshape(-1)[vocab[tk]] in (1.,2.)
		assert len(vocab) == 5
		print('Test 5 passed!')
	except AssertionError:
		print('Test 5 failed: las frecuencias no están bien contadas')


if __name__ == '__main__':
	test1()
	test2()
	test3()
	test4()
	test5()