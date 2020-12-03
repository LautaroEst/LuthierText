import numpy as np


# Reweight methods:

def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = np.zeros(df.shape)
    mask = expected != 0
    #np.putmask(oe, mask, df[mask] / expected[mask])
    #oe[mask].reshape(1,-1) = (df[mask] / expected[mask]).reshape(-1)
    oe[mask] = df[mask] / expected[mask]
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df


def tfidf(df):
    # Inverse document frequencies:
    doccount = float(df.shape[1])
    freqs = df.astype(bool).sum(axis=1)
    idfs = np.log(doccount / freqs)
    idfs[np.isinf(idfs)] = 0.0  # log(0) = 0
    # Term frequencies:
    col_totals = df.sum(axis=0)
    tfs = np.zeros_like(df)
    mask = col_totals!=0 
    tfs[:,mask] = df[:,mask] / col_totals[mask]
    return (tfs.T * idfs).T


