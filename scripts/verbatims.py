import elasticsearch
import logging
import numpy as np
import os
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.dataset as ds
import sys

from argparse import ArgumentParser, Namespace
from bs4 import BeautifulSoup
from collections import defaultdict
from elasticsearch import RequestError
from elasticsearch_dsl import Search
from nltk.tokenize import sent_tokenize
from pickle import PicklingError
from typing import Any, DefaultDict, List, Tuple

from courtlistener import opinions_in_arrowbatch
from utils import OpinionDocument, OpinionSentence
from utils import batch_iterator, config, elastic_init, queue_worker, make_clean_folder, \
    multiprocess, parquet_dataset_iterator, random_name

config()
elasticsearch.logger.setLevel(logging.WARNING)
_args: Namespace = Namespace()

# Deeper recursion for BeautifulSoup
max_recursion = 100000
sys.setrecursionlimit(max_recursion)


def index_all_opinions():
    """
    Adds all opinions to an index.

    :return:
    """
    @queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        elastic_init(_args.env)

        for opinion in opinions_in_arrowbatch(x):
            OpinionDocument(opinion_id=opinion.opinion_id,
                            raw_text=opinion.raw_text).save()
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=8)
    multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description=f'Add opinions to index {OpinionDocument.Index.name}')


def index_all_sentences():
    """
    Adds all sentences of each opinion to an index.

    :return:
    """
    @queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        elastic_init(_args.env)

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
                    logging.debug(f"opinion_id={opinion.opinion_id}, sentence_id={sentence_id}, "
                                  f"raw_text={sentence}, Error={repr(e)}")
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=8)
    multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description=f'Add opinions to index {OpinionSentence.Index.name}')


def _create_search_verbatim(cited_opinion_id: int, full: str) -> Search:
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
    return OpinionDocument.search(). \
        query("match", opinion_id=cited_opinion_id). \
        query("intervals", raw_text=raw_text_query)


def search_verbatims():
    """
    From all the potential verbatim quotes, identify the true citations that were made through a verbatim quote
    Using Elastic Search with the quote.

    :return:
    """
    @queue_worker
    def _search(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        elastic_init(_args.env)

        data = []
        batch: pd.DataFrame = x.to_pandas()
        for _, d in batch.iterrows():
            d: pd.Series
            cited_opinion_id = d['cited_opinion_id']
            full = d['verbatim']

            # STEP 1 - YES or NO: is the verbatim from the cited opinion ??
            base_search = _create_search_verbatim(cited_opinion_id, full)
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

        # Save the Dataset for YES/NO as PARQ
        if _args.save and len(data) > 0:
            file_out = os.path.join(_args.dest, f'{random_name()}.parq')
            pd.DataFrame(data).to_parquet(file_out)

        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    logging.info(f'Destination {_args.dest}')
    make_clean_folder(_args.dest)
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=128)
    multiprocess(worker_fn=_search, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description='Search Verbatims')


def search_verbatim_sentence():
    """
    From all the verbatim citations, identify which sentence was cited from the cited opinion.
    This does not generate any dataset, it updated the ElasticSearch index of OpinionSentence.

    :return:
    """
    data: DefaultDict[str, int] = defaultdict(lambda: 0)

    @queue_worker
    def _search_sentence(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        elastic_init(_args.env)

        batch: pd.DataFrame = x.to_pandas()
        for _, d in batch.iterrows():
            d: pd.Series
            cited_opinion_id = d['cited_opinion_id']
            full = d['verbatim']

            # Based on the same search as above
            base_search = _create_search_verbatim(cited_opinion_id, full)

            # Use HIGHLIGHT with the previous search to identify WHERE the matching occurs
            verbatim_search = base_search. \
                highlight("raw_text",
                          type="unified",
                          fragment_size=1000,
                          order="score",
                          number_of_fragments=1). \
                params(filter_path=['hits.hits.highlight'])

            try:
                results = verbatim_search.execute()
                highlight = results[0].meta['highlight']['raw_text'][0]
            except (RequestError, KeyError, IndexError):
                continue

            # The highlight with highest score is returned
            # It is a text with more than one sentence
            # We select the sentence that includes the highest number of highlighted terms (<em> in Highlight HTML)
            highlight_sentences = map(lambda s: s if "<em>" in s else "", sent_tokenize(highlight))
            soups = [BeautifulSoup(s, "html.parser") for s in highlight_sentences]

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
                query("intervals", raw_text=sentence_query). \
                params(filter_path=["hits.hits._id"])

            try:
                results = sentence_search.execute()
                sentence_elastic_id = results[0].meta["id"]
            except (RequestError, KeyError, IndexError):
                continue

            data[sentence_elastic_id] += 1

        return x.num_rows

    @queue_worker
    def _update_sentences(x: List[Tuple[str, int]]) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        elastic_init(_args.env)

        for elastic_id, count_citations in x:
            s = OpinionSentence()
            s.meta.id = elastic_id
            s.update(highlight=True, count_citations=count_citations)
        return len(x)

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=128)
    multiprocess(worker_fn=_search_sentence, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description='Search Verbatims Sentences')

    logging.info(f"Done with collecting {len(data)} updates.")
    try:
        pickle.dump(dict(data), open('data.pickle', 'wb'))
    except PicklingError:
        logging.info(f"Could not save data into ./data.pickle")
    logging.info(f"Saving data in file ./data.pickle")
    logging.info(f"Updating collection.")
    iterator, nb_rows = batch_iterator(list(data.items()), batch_size=128)
    multiprocess(worker_fn=_update_sentences, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description="Update Sentences Collection")


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

    parser_index = subparsers.add_parser(name='index-sentences', description='Add each sentence of each opinion '
                                                                             'to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_sentences)

    parser_search = subparsers.add_parser(name='search-verbatims', description='Search the verbatim quotes.')
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

    parser_search = subparsers.add_parser(name='search-sentences', description='Search the exact sentence the verbatim '
                                                                               'quotes originate from.')
    parser_search.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_search.add_argument('--env', type=str, help='Path to the env file')
    parser_search.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_search.set_defaults(func=search_verbatim_sentence)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()
    logging.info('Done.')
    sys.exit()


if __name__ == '__main__':
    main()
