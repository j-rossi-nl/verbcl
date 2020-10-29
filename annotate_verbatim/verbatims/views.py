import pandas as pd
import pyarrow.dataset as ds
import numpy as np
import os

from django.shortcuts import render
from filelock import FileLock

from courtlistener import Opinion
from .apps import VerbatimsConfig
# Create your views here.

core: pd.DataFrame = ds.dataset(VerbatimsConfig.core_path).to_table().to_pandas().set_index('opinion_id')
cited: pd.DataFrame = ds.dataset(VerbatimsConfig.cited_path).to_table().to_pandas().set_index('opinion_id')

assert core.index.is_unique
assert cited.index.is_unique


def _find_sample():
    while True:
        random_core = core.sample(1)

        op = Opinion(opinion_id=random_core.index[0], opinion_html=random_core.iloc[0]['html_with_citations'])
        try:
            vs = list(op.verbatim(max_words_before_after=100, min_words_verbatim=5))
            v = np.random.choice(vs, size=1)[0]
            cit = cited.loc[v['cited_opinion_id']]
        except (KeyError, ValueError):
            continue

        return v, Opinion(v['cited_opinion_id'], cit['html_with_citations'])


current_verbatim = {}


def index(request):
    global current_verbatim

    if request.method == 'POST':
        # After click
        names = ['next', 'yes', 'no']
        button_name = names[[n in request.POST for n in names].index(True)]

        if button_name in ['yes', 'no']:
            annotation = {
                'citing_opinion_id': current_verbatim['citing_opinion_id'],
                'cited_opinion_id': current_verbatim['cited_opinion_id'],
                'snippet': current_verbatim['snippet'],
                'verbatim': current_verbatim['verbatim'],
                'is_verbatim': button_name == 'yes'
            }

            path = '/home/juju/PycharmProjects/courtlistener/data/verbatim_annotations.json'
            with FileLock(path + '.lock'):
                if os.path.exists(path):
                    df = pd.read_json(path, orient='records', lines=True)
                    df = df.append(annotation, ignore_index=True)
                else:
                    df = pd.DataFrame([annotation])
                df.to_json(path, orient='records', lines=True)

    # Whatever the case, find a next sample
    v, cit = _find_sample()
    current_verbatim = v
    begin, end = v['span_in_snippet']
    cit: Opinion
    context = {
        'snippet_html': v['snippet'][:begin] + '<b>"' + v['verbatim'] + '"</b>' + v['snippet'][end:],
        'verbatim': v['verbatim'],
        'cited_opinion': str(cit.soup)
    }

    return render(request=request, template_name='verbatim.html', context=context)
