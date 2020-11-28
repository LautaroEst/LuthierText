from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
from ..count import count_bag_of_ngrams, reduce_by_freq


def _filter_by_frequency(X, full_vocab, reshape_X_fn, get_freqs_fn, 
min_count=0., max_count=None, max_words=None):
    """ Función privada para limitar tokens por frecuencia. Modifica
    tanto el vocabulario como la matriz de coocurrencias que se le pasa. """

    freqs = get_freqs_fn(X)
    sorted_indices = np.argsort(freqs)[::-1]
    sorted_frequencies = freqs[sorted_indices]

    if min_count <= 0 and max_count is None:
        mask = np.ones(X.shape[0],dtype=bool)

    else:
        if max_count is None:
            max_count = np.inf
        mask = np.logical_and(sorted_frequencies <= max_count, 
                                sorted_frequencies >= min_count)

    sorted_indices = sorted_indices[mask]
    
    if max_words is not None:
        sorted_indices = sorted_indices[:max_words]
    
    X = reshape_X_fn(X,sorted_indices)
    idx_to_tk = {idx:tk for tk,idx in full_vocab.items()}
    vocab = {idx_to_tk[idx]:i for i,idx in enumerate(sorted_indices)}



    """ keys = np.array(list(full_vocab.keys()))
    values = np.array(list(full_vocab.values()))
    keys = keys[np.argsort(values)]

    freqs = get_freqs_fn(X)
    sorted_idx = np.argsort(freqs)[::-1]
    sorted_freqs = freqs[sorted_idx]
    X = reshape_X_fn(X,sorted_idx)
    keys = keys[sorted_idx]

    if min_count <= 0 and max_count is None:
        pass

    else:
        if max_count is None:
            max_count = np.inf

        mask = np.logical_and(sorted_freqs <= max_count, sorted_freqs >= min_count)
        X = reshape_X_fn(X,mask)
        keys = keys[mask]

    if max_words is not None:
        mask = np.arange(X.shape[0]) < max_words
        X = reshape_X_fn(X,mask)
        keys = keys[:max_words]

    vocab = {tk:idx for idx,tk in enumerate(keys)} """

    return X, vocab



def _filter_by_token(cooccurrences_dict, tokens=None, negative=True):
    """ Función privada para eliminar los pares de palabras que contienen palabras
    no deseadas. Si negative es False, se devuelve un nuevo diccionario que contiene
    los pares de palabras formados por tokens de la lista tokens. Si es True, 
    devuelve todos los tokens que aparecieron, menos los de la lista tokens."""
    if tokens is None:
        return cooccurrences_dict
    
    if negative:
        new_dict = {key:val for key, val in cooccurrences_dict.items() if key[0] not in tokens and key[1] not in tokens}
    else:
        new_dict = {key:val for key, val in cooccurrences_dict.items() if key[0] in tokens and key[1] in tokens}
    return new_dict


def _check_tokenizer(tokenizer):
    """ Función privada para checkear el tokenizer """

    if tokenizer is None:
        tokenizer = lambda x: x
    elif not callable(tokenizer):
        raise ValueError('Tokenizer must be callable or None.')

    return tokenizer


def _check_context_size_and_window(window,left_n,right_n):
    """ Función privada para checkear contexto y ventana """

    if (isinstance(left_n,int) and left_n > 0) and (isinstance(right_n,int) and right_n > 0):
        pass
    else:
        raise ValueError('left_n and rigth_n must be integers greater than 0.')
    
    if window is None:
        window = [1. for i in range(left_n+right_n+1)]
        #window[left_n] = 0.
    else:
        try:
            if len(window) != left_n + right_n + 1:
                raise ValueError('Size of window must match the size of context (left_n+right_n+1).')
        except TypeError:
            raise ValueError('window must be either an array-like object with the lenght of context, or None.')

    return window


def word_by_word_cooccurrence(corpus, window=None, left_n=2, right_n=2, 
    tokenizer=None, min_count=0., max_count=None, max_words=None):
    """ Devuelve una matriz de coocurrencias entre palabras y una ventana 
    de palabras cercanas. Las filas son las palabras y las columnas, los 
    features. Es decir, es una matriz cuadrada. """

    # Se checkea si tokenizer, window, left_n, y right_n son válidos:
    tokenizer = _check_tokenizer(tokenizer)
    window = _check_context_size_and_window(window,left_n,right_n)
    
    # Se define el vocabulario completo pero vacío:
    cooccurrences_dict = defaultdict(float)
    full_vocab = defaultdict()
    full_vocab.default_factory = full_vocab.__len__

    # Se cuentan todas las apariciones de las palabras y en qué contexto
    # apareció cada una de ellas:
    unk_idx = -1
    for doc in tqdm(corpus):
        indices = [full_vocab[tk] for tk in tokenizer(doc)]
        for i in range(left_n):
            indices.insert(0,unk_idx)
        for i in range(right_n):
            indices.append(unk_idx)
        for i, idx in enumerate(indices):
            context = indices[i-left_n:i+right_n+1]
            for j, c in zip(window,context):
                cooccurrences_dict[(idx, c)] += j

    cooccurrences_dict = _filter_by_token(dict(cooccurrences_dict), [unk_idx], negative=True)
    full_vocab = dict(full_vocab)
    i, j = zip(*cooccurrences_dict.keys())
    data = list(cooccurrences_dict.values())
    vocab_len = len(full_vocab)
    X = coo_matrix((data, (i,j)),shape=(vocab_len,vocab_len)).tocsr()

    # Limito por frecuencia o por tope máximo de palabras
    def get_freqs(X):
        return np.array(X.diagonal()).reshape(-1)

    def reshape_X(X,mask_or_indices):
        X = X[mask_or_indices,:]
        X = X[:,mask_or_indices]
        return X

    X, vocab = _filter_by_frequency(X,full_vocab,reshape_X,get_freqs,min_count,
                                    max_count,max_words)

    return X, vocab


def word_by_document_cooccurrence(corpus, tokenizer=None, 
    min_count=0., max_count=None, max_words=None):
    """
    Devuelve la matriz de coocurrencias entre palabras y documentos. Cada fila es una palabra
    y cada columna, un documento distinto.
    """
    X, vocab = count_bag_of_ngrams(corpus, ngram_range=(1,1), tokenizer=tokenizer)
    return reduce_by_freq(X.T.tocsr(), vocab, min_count, max_count, max_words)


def word_by_category_cooccurrence(corpus, labels, tokenizer=None,
    min_count=0., max_count=None, max_words=None):
    """
    Devuelve la matriz de coocurrencias entre palabras y la categoría a la que pertence 
    el documento. Es decir, las filas de la matriz son las palabras y las columnas son
    todas las categorías posibles, y todas las entradas de la matriz contienen la cuenta
    de cuántas veces apareció la palabra en un documento de cada categoría.
    """
    categories = sorted(set(labels)) # Se asume que los labels son 0, 1, ..., len(categories)
    cooccurrences_dict = defaultdict(float)
    full_vocab = defaultdict()
    full_vocab.default_factory = full_vocab.__len__

    if tokenizer is None:
        tokenizer = lambda x: x

    for doc, label in zip(corpus, labels):
        for tk in tokenizer(doc):
            cooccurrences_dict[(full_vocab[tk],label)] += 1.

    full_vocab = dict(full_vocab)
    i, j = zip(*cooccurrences_dict.keys())
    data = list(cooccurrences_dict.values())
    X = coo_matrix((data, (i,j)),shape=(len(full_vocab),len(categories)))
    return reduce_by_freq(X.tocsr(), full_vocab, min_count, max_count, max_words)

