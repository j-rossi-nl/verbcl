import elasticsearch
import logging
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import sys
import utils

from argparse import ArgumentParser, Namespace
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from elasticsearch import RequestError
from elasticsearch_dsl import analyzer, connections
from elasticsearch_dsl import Boolean, Document, Integer, Text
from nltk.tokenize import sent_tokenize
from typing import Any

from courtlistener import opinions_in_arrowbatch

utils.config()
elasticsearch.logger.setLevel(logging.WARNING)
_args: Namespace = Namespace()

# Deeper recursion for BeautifulSoup
max_recursion = 100000
sys.setrecursionlimit(max_recursion)


class OpinionDocument(Document):
    opinion_id = Integer()
    raw_text = Text(analyzer=analyzer('alpha_stop_stem',
                                      type='custom',
                                      tokenizer='classic',
                                      filter=['lowercase', 'asciifolding', 'stop', 'snowball']))

    class Index:
        name = 'juju-01'


class OpinionSentence(Document):
    opinion_id = Integer()
    sentence_id = Integer()
    highlight = Boolean()
    count_citations = Integer()
    raw_text = Text(analyzer=analyzer('alpha_stop_stem',
                                      type='custom',
                                      tokenizer='classic',
                                      filter=['lowercase', 'asciifolding', 'stop', 'snowball']))

    class Index:
        name = 'juju-02'

    # Overloading save() to set False as the default value for highlight
    # See https://stackoverflow.com/questions/52089760/elasticsearch-dsl-lib-set-default-value-for-text-field-on-saving-document
    def save(self, **kwargs):
        if self.highlight is None:
            self.highlight = False
        if self.count_citations is None:
            self.count_citations = 0

        return super().save(**kwargs)


def index_all_opinions():
    """
    Adds all opinions to an index.

    :return:
    """
    @utils.queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        for opinion in opinions_in_arrowbatch(x):
            OpinionDocument(opinion_id=opinion.opinion_id,
                            raw_text=opinion.raw_text).save()
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = utils.parquet_dataset_iterator(dataset=dataset, batch_size=8)
    utils.multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                       nb_workers=_args.num_workers, description=f'Add opinions to index {OpinionDocument.Index.name}')


def index_all_sentences():
    """
    Adds all sentences of each opinion to an index.

    :return:
    """
    @utils.queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        for opinion in opinions_in_arrowbatch(x):
            for sentence_id, sentence in enumerate(sent_tokenize(opinion.raw_text)):
                try:
                    OpinionSentence(
                        opinion_id=opinion.opinion_id,
                        sentence_id=sentence_id,
                        raw_text=sentence,
                        highlight=False,
                        count_citations=0
                    ).save()
                except RequestError as e:
                    logging.debug(f"opinion_id={opinion.opinion_id}, sentence_id={sentence_id}, raw_text={sentence}, Error={repr(e)}")
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = utils.parquet_dataset_iterator(dataset=dataset, batch_size=8)
    utils.multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                       nb_workers=_args.num_workers, description=f'Add opinions to index {OpinionSentence.Index.name}')


def search_verbatims():
    """
    From all the potential verbatim quotes, identify the true citations that were made through a verbatim quote
    Using Elastic Search with the quote.

    :return:
    """
    @utils.queue_worker
    def _search(x: pa.RecordBatch) -> int:
        data = []
        batch: pd.DataFrame = x.to_pandas()
        for _, d in batch.iterrows():
            d: pd.Series
            cited_opinion_id = d['cited_opinion_id']
            full = d['verbatim']
            first_10 = ' '.join(full.split(' ')[:10])
            last_10 = ' '.join(full.split(' ')[-10:])

            # STEP 1 - YES or NO: is the verbatim from the cited opinion ??
            # Using INTERVALS query
            # Looking for the whole verbatim, the first 10 words, the last 10 words
            raw_text_query = {
                'any_of': {
                    'intervals': [
                        {'match': {'query': full, 'ordered': True, 'max_gaps': 10}},
                        {'match': {'query': first_10, 'ordered': True, 'max_gaps': 2}},
                        {'match': {'query': last_10, 'ordered': True, 'max_gaps': 2}}
                    ]
                }
            }
            base_search = OpinionDocument.search(). \
                query("match", opinion_id=cited_opinion_id). \
                query("intervals", raw_text=raw_text_query)

            score_search = base_search.params(filter_path=['hits.hits._score'])
            try:
                results = score_search.execute()
                score = results[0].meta.score
            except (RequestError, KeyError, IndexError):
                # The search did not return any result at all
                # The PARAMS restrain the query to ONE document, and the INTERVALS queries will dismiss
                # documents that do not match the rules.
                # So this happens when the alledged verbatim does not appear in the cited opinion
                score = -1
            except:
                score = -1

            data.append({**(d.to_dict()), 'score': float(score)})
            if score == -1:
                continue

            # STEP 2 - If YES, which sentence was it ?
            # Use HIGHLIGHT with the previous search to identify WHERE the matching occurs
            verbatim_search = score_search.highlight("raw_text",
                                                     type="unified",
                                                     fragment_size=1000,
                                                     order="score",
                                                     number_of_fragments=1)

            try:
                results = verbatim_search.execute()
                highlight = results[0].meta['highlight']['raw_text'][0]
            except (RequestError, KeyError, IndexError):
                continue

            # The highlight with highest score is returned
            # It is a text with more than one sentence
            # We select the sentence that includes the highest number of highlighted terms (<em> in Highlight HTML)
            highlight_sentences = sent_tokenize(highlight)
            soups = [BeautifulSoup(s, "html") for s in highlight_sentences]

            count_em = [len(s.find_all("em")) for s in soups]
            best_index = np.argmax(count_em)
            best_text = soups[best_index].text

            # Use the database of opinion sentences to find the ID of this sentence in the cited opinion
            sentence_query = {
                "all_of": {
                    "intervals": [
                        {"match": {"query": best_text, "ordered": True, "max_gaps": 1}}
                    ]
                }
            }

            sentence_search = OpinionSentence.search(). \
                query("match", opinion_id=cited_opinion_id). \
                query("intervals", raw_text=sentence_query)

            try:
                results = sentence_search.execute()
                sentence_elastic_id = results[0].meta['id']
            except (RequestError, KeyError, IndexError):
                continue

            # From the Elasticsearch ID, we retrieve the sentence object and its sentence_id within the document
            # We annotate it as TRUE (it is a highlight)
            # We save the modification
            sentence = OpinionSentence.get(id=sentence_elastic_id)
            sentence.highlight = True
            sentence.count_citations += 1
            sentence.save()

        # Save the Dataset for YES/NO as PARQ
        if _args.save and len(data) > 0:
            file_out = os.path.join(_args.dest, f'{utils.random_name()}.parq')
            pd.DataFrame(data).to_parquet(file_out)

        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    logging.info(f'Destination {_args.dest}')
    utils.make_clean_folder(_args.dest)
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = utils.parquet_dataset_iterator(dataset=dataset, batch_size=128)
    utils.multiprocess(worker_fn=_search, input_iterator_fn=iterator, total=nb_rows,
                       nb_workers=_args.num_workers, description='Search Verbatims')


def parse_args():
    argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    parser_index = subparsers.add_parser(name='index-opinions', description='Add all opinions to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_opinions)

    parser_index = subparsers.add_parser(name='index-sentences', description='Add each sentence of each opinion to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_sentences)

    parser_search = subparsers.add_parser(name='search', description='Search the verbatim quotes.')
    parser_search.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_search.add_argument('--dest', type=str, help='Destination Path for the PARQUET dataset')
    parser_search.add_argument('--max-words-extract', type=int, default=100, help='Number of words to consider around '
                                                                                  'a citation')
    parser_search.add_argument('--min-verbatim', type=int, default=5, help='Minimum number of words between quotes to'
                                                                           'consider it as a verbatim quote')
    parser_search.add_argument('--save', default=False, action='store_true', help='Save the YES/NO in Destination')
    parser_search.add_argument('--env', type=str, help='Path to the env file')
    parser_search.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_search.set_defaults(func=search_verbatims)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()

    # Initialize the connection to Elasticsearch
    # Making use of elasticsearch_dsl persistence features
    load_dotenv(os.path.expanduser(_args.env))
    if os.getenv('ELASTIC_CLOUD_ID') is not None:
        connections.create_connection(cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
                                      api_key=(
                                          os.getenv('ELASTIC_CLOUD_API_ID'), os.getenv('ELASTIC_CLOUD_API_KEY')),
                                      timeout=1000)
    else:
        connections.create_connection(host=os.getenv('ELASTIC_HOST'),
                                      port=os.getenv('ELASTIC_PORT'),
                                      timeout=1000,
                                      maxsize=40)

    OpinionDocument.init()
    OpinionSentence.init()

    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
