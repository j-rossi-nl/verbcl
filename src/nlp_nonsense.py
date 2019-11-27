#!/usr/bin/env python
# coding: utf-8

# based on
# http://cs229.stanford.edu/proj2014/Ian%20Tenney,%20A%20General-Purpose%20Sentence-Level%20Nonsense%20Detector.pdf
# and https://github.com/iftenney/nlp-nonsense

import en_core_web_sm  # or en_core_web_lg if need tokenization.
import pandas as pd
import re

from spacy.tokens import Doc
from typing import List, Dict, Any

_nlp = en_core_web_sm.load()
_x_v_y = re.compile(r'^[A-Z][\w\s]+ v. [A-Z][\w\s]+')


def _make_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic features."""

    df['f_nwords'] = df['word'].map(len)

    def pos_counter(s: str, tags: List[str]) -> int:
        return sum(1 for x in s if x in tags)

    def punct_counter(s: str) -> int:
        return pos_counter(s, ['PUNCT'])
    df['f_npunct'] = df['pos'].map(punct_counter)

    def verb_counter(s: str) -> int:
        return pos_counter(s, ['VERB'])
    df['f_nverb'] = df['pos'].map(verb_counter)

    # fraction named entities recognized (ner) -- 'O' is not recognized
    df['f_nner'] = df['ner'].map(lambda ts: sum(1 for t in ts
                                                if t != 'O'))

    # Check standard sentence pattern:
    # We might not have the entire sentence, so we check only for the beginning
    def check_sentence_pattern(s: str) -> bool:
        ss = s.strip()
        # Has many words, starts with some characters, or Uppercase
        first_check = (len(ss) > 0) and (ss[0] in r'`“”"\'' or ss[0].isupper())

        # We want to get rid of 'G v. R.' that are often considered as sentences
        second_check = _x_v_y.match(s) is None

        return first_check and second_check

    df['f_sentence_pattern'] = df['__TEXT__'].map(check_sentence_pattern)

    return df


def _build_row(doc: Doc) -> Dict[str, Any]:
    row = {
        '__TEXT__': str(doc),
        'pos': [i.pos_ for i in doc],
        'ner': [e.label for e in doc.ents],
        'word': [i.text for i in doc]
    }
    return row


def _preprocess_spacy(spacy_nlp: List[Doc]) -> pd.DataFrame:
    """
    Expect the following pipes applied : tagger, ner

    :param spacy_nlp:
    :return:
    """
    data = [_build_row(doc) for doc in spacy_nlp]
    df = pd.DataFrame(data)
    return df


def _do_sentence_or_not(df: pd.DataFrame) -> List[bool]:
    features = _make_basic_features(df)
    results = []
    for _, row in features.iterrows():
        results.append(row['f_sentence_pattern'] and (row['f_npunct'] + row['f_nwords']) > 5 and row['f_nner'] > 0
                       and row['f_nverb'] > 0)
    return results


def sentence_or_not_spacy(spacy_nlp: List[Doc]) -> List[bool]:
    return _do_sentence_or_not(_preprocess_spacy(spacy_nlp))
