import sys
import os
import logging
import logging.config
import functools
import elasticsearch
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd

import utils

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
from typing import Any

utils.config()
elasticsearch.logger.setLevel(logging.WARNING)
_args: Namespace = Namespace()


def elastic_function(func):
    """
    Decorator for a function connecting to Elastic Search.

    :param func: signature (connection: Elasticsearch) -> None: uses a connection
    :return: a function with no argument
    """

    @functools.wraps(func)
    def wrapper():
        """

        :return:
        """
        load_dotenv(_args.env)
        if os.getenv('CLOUD_ID') is not None:
            connection = Elasticsearch(cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
                                       api_key=(os.getenv('ELASTIC_CLOUD_API_ID'), os.getenv('ELASTIC_CLOUD_API_KEY')),
                                       timeout=1000)
        else:
            connection = Elasticsearch(host=os.getenv('ELASTIC_HOST'), port=os.getenv('ELASTIC_PORT'),
                                       timeout=1000)

        func(connection)

    return wrapper


@elastic_function
def index_all_opinions(es: Elasticsearch):
    """
    Adds all opinions to an index.

    :return:
    """

    @utils.queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        for opinion in utils.opinions_in_arrowbatch(x):
            es.index(
                index=_args.index,
                body={
                    'opinion_id': opinion.opinion_id,
                    'raw_text': opinion.raw_text
                }
            )
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = utils.parquet_dataset_iterator(dataset=dataset, batch_size=8)
    utils.multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                       nb_workers=_args.num_workers, description=f'Add opinions to index {_args.index}')


@elastic_function
def search_verbatims(es: Elasticsearch):
    """
    From all the potential verbatim quotes, identify the true citations that were made through a verbatim quote
    Using Elastic Search with the quote.

    :param es:
    :return:
    """
    @utils.queue_worker
    def _search(x: pa.RecordBatch) -> int:
        data = []
        for opinion in utils.opinions_in_arrowbatch(x):
            for verbatim in opinion.verbatim():
                s = Search(index=_args.index). \
                    using(es). \
                    query("match", opinion_id=verbatim['cited_opinion_id']). \
                    query("match", raw_text=verbatim['verbatim']). \
                    params(filter_path=['hits.hits._score'])
                results = s.execute()
                try:
                    score = results[0].meta.score
                except KeyError:
                    score = -1

                s = Search(index=_args.index). \
                    using(es). \
                    query("match", raw_text=verbatim['verbatim']). \
                    extra(from_=0, size=100). \
                    params(filter_path=['hits.hits._source.opinion_id'])
                results = s.execute()
                try:
                    op_ids = [r.opinion_id for r in results]
                    rank = -1 if verbatim['cited_opinion_id'] not in op_ids else op_ids.index(
                        verbatim['cited_opinion_id']) + 1
                except KeyError:
                    rank = -1

                data.append({**verbatim, 'score': float(score), 'rank': int(rank)})

        if len(data) > 0:
            file_out = os.path.join(_args.dest, f'{utils.random_name()}.parq')
            pd.DataFrame(data).to_parquet(file_out)

        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = utils.parquet_dataset_iterator(dataset=dataset, batch_size=8)
#    for batch in iterator():
#        _search(batch)
    utils.multiprocess(worker_fn=_search, input_iterator_fn=iterator, total=nb_rows,
                       nb_workers=_args.num_workers, description='Search Verbatims')


def parse_args():
    argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    parser_index = subparsers.add_parser(name='index', description='Add all opinions to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--index', type=str, help='Index name')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_opinions)

    parser_search = subparsers.add_parser(name='search', description='Search the verbatim quotes.')
    parser_search.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_search.add_argument('--dest', type=str, help='Destination Path for the PARQUET dataset')
    parser_search.add_argument('--index', type=str, help='Name of the ElasticSearch index where the opinions'
                                                         'have been added')
    parser_search.add_argument('--min-score', type=int, default=100, help='Minimum score to consider a match')
    parser_search.add_argument('--env', type=str, help='Path to the env file')
    parser_search.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_search.set_defaults(func=search_verbatims)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
