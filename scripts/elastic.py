import elasticsearch
import logging
import os
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import sys
import time

import utils

from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
from elasticsearch import RequestError
from elasticsearch_dsl import analyzer, connections, Document, Integer, Text
from multiprocessing import current_process
from typing import Any

from courtlistener import opinions_in_arrowbatch

utils.config()
elasticsearch.logger.setLevel(logging.WARNING)
_args: Namespace = Namespace()


class OpinionDocument(Document):
    opinion_id = Integer()
    raw_text = Text(analyzer=analyzer('alpha_stop_stem',
                                      type='custom',
                                      tokenizer='classic',
                                      filter=['lowercase', 'asciifolding', 'stop', 'snowball']))

    class Index:
        name = 'juju-01'


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
            full = d['verbatim']
            first_10 = ' '.join(full.split(' ')[:10])
            last_10 = ' '.join(full.split(' ')[-10:])

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
            s = OpinionDocument.search(). \
                query("match", opinion_id=d['cited_opinion_id']). \
                query("intervals", raw_text=raw_text_query). \
                params(filter_path=['hits.hits._score'])
            try:
                results = s.execute()
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

        if len(data) > 0:
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

    parser_index = subparsers.add_parser(name='index', description='Add all opinions to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_opinions)

    parser_search = subparsers.add_parser(name='search', description='Search the verbatim quotes.')
    parser_search.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_search.add_argument('--dest', type=str, help='Destination Path for the PARQUET dataset')
    parser_search.add_argument('--max-words-extract', type=int, default=100, help='Number of words to consider around '
                                                                                  'a citation')
    parser_search.add_argument('--min-verbatim', type=int, default=5, help='Minimum number of words between quotes to'
                                                                           'consider it as a verbatim quote')
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
                                      timeout=1000)
    OpinionDocument.init()

    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
