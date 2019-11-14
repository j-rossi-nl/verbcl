#!/usr/bin/env python
# coding: utf-8

# based on http://cs229.stanford.edu/proj2014/Ian%20Tenney,%20A%20General-Purpose%20Sentence-Level%20Nonsense%20Detector.pdf
# and https://github.com/iftenney/nlp-nonsense


import sys, os, re, json
import spacy
import en_core_web_sm # or en_core_web_lg if need tokenization.
import itertools
import pandas as pd


from collections import Counter, OrderedDict
from numpy import *

nlp = en_core_web_sm.load()


def _make_basic_features(df):
    """Compute basic features."""

    df['f_nchars'] = df['__TEXT__'].map(len)
    df['f_nwords'] = df['word'].map(len)

    punct_counter = lambda s: sum(1 for c in s
                                  if (not c.isalnum())
                                      and not c in
                                        [" ", "\t"])
    df['f_npunct'] = df['__TEXT__'].map(punct_counter)
    df['f_rpunct'] = df['f_npunct'] / df['f_nchars']

    df['f_ndigit'] = df['__TEXT__'].map(lambda s: sum(1 for c in s
                                  if c.isdigit()))
    df['f_rdigit'] = df['f_ndigit'] / df['f_nchars']

    upper_counter = lambda s: sum(1 for c in s if c.isupper())
    df['f_nupper'] = df['__TEXT__'].map(upper_counter)
    df['f_rupper'] = df['f_nupper'] / df['f_nchars']

    # fraction named entities recognized (ner) -- 'O' is not recognized
    df['f_nner'] = df['ner'].map(lambda ts: sum(1 for t in ts
                                              if t != 'O'))
    df['f_rner'] = df['f_nner'] / df['f_nwords']

    # Check standard sentence pattern:
    # if starts with capital, ends with .?!
    def check_sentence_pattern(s):
        ss = s.strip(r"""`"'""").strip()
        return s[0].isupper() and (s[-1] in '.?!\n')
    df['f_sentence_pattern'] = df['__TEXT__'].map(check_sentence_pattern)

    # Normalize any LM features
    # by dividing logscore by number of words
    lm_cols = {c:re.sub("_lmscore_", "_lmscore_norm_",c)
               for c in df.columns if c.startswith("f_lmscore")}
    for c,cnew in lm_cols.items():
        df[cnew] = df[c] / df['f_nwords']

    return df


def _preprocess(raw):
    data = []
    for idx,text in enumerate(raw):
        doc = nlp(text)
        row = {"__TEXT__": text}
        row['ner'] = [i.pos_ for i in doc]
        row['sentiment'] = doc.sentiment
        row['word'] = [i.text for i in doc]
        data.append(row)

    df = pd.DataFrame(data)
    return df


def sentence_or_not(texts):
    df = _preprocess(texts)
    features = _make_basic_features(df)

    results = []
    for _,row in features.iterrows():
        results.append(row['f_sentence_pattern'] and (row['f_npunct'] + row['f_nwords']) > 5 and row['f_nner'] > 0)
    return results
