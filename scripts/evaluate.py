import logging
import os
import pandas as pd
import sys
import utils

from argparse import ArgumentParser, Namespace
from operator import itemgetter
from pymongo.cursor import Cursor
from pymongo import MongoClient
from pyspin.spin import make_spin, Default
from rouge import Rouge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Callable, Dict, List, Tuple

# Configure
utils.config()

# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args: Namespace = Namespace()


def evaluate_summaries():
    @utils.queue_worker
    def worker_evaluate_ngram_cosine(x: List[Dict[str, Any]]) -> int:
        """
        Get a list of tuple Summary / Ground and generate comparison metrics.
        Works on merged dataset.

        :param x: batch of data
        :return:
        """
        vct = CountVectorizer(
            stop_words='english',
            ngram_range=(3, 4)
        )

        data = []
        for sample in x:
            logging.debug(f'Processing Opinion ID: {sample["opinion_id"]}')
            evaluated_summary = sample[_args.eval_field]
            grounds = sample.get('verbatims', None)
            if grounds is None or len(grounds) == 0:
                # No ground truth to compare to
                continue

            # Evaluation: cosine similarity with 4-grams
            # If one the sentences in the ground truth is in the summary, it will be detected
            corpus = [evaluated_summary] + grounds
            try:
                vct.fit(grounds)
                bows = vct.transform(corpus)
                sims = cosine_similarity(bows)

                evaluation_result = max(sims[0, 1:])
                data.append({'opinion_id': sample['opinion_id'], 'eval_score': evaluation_result})
            except ValueError:
                # Empty vocabulary
                continue

        if len(data) > 0:
            file_name = os.path.join(_args.dest, f'{utils.random_name()}.parq')
            pd.DataFrame(data).to_parquet(file_name)

        return len(x)

    @utils.queue_worker
    def worker_evaluate_rouge(x: List[Dict[str, Any]]) -> int:
        """

        :param x:
        :return:
        """
        # Restrict to the samples that have verbatims (ie opinion that were cited with verbatims)
        valid_samples = [s for s in x if 'verbatims' in s]
        if len(valid_samples) == 0:
            return len(x)

        evaluator = Rouge(
            metrics=['rouge-n', 'rouge-l', 'rouge-w'],
            max_n=4,
            apply_avg=False,
            apply_best=False
        )

        refs = [' '.join(s['verbatims']) for s in valid_samples]
        hyps = [s[_args.eval_field] for s in valid_samples]
        scores = evaluator.get_scores(hypothesis=hyps, references=refs)
        # scores is shaped as:
        # scores = {'rouge-1': [{'f': [0.1], 'r': [0.2], 'p': [0.3]}, ...] N elts = N references,
        #           'rouge-2': [{'f': [0.4], 'r': [0.5], 'p': [0.6]}, ...],

        data = {k: {} for k in map(itemgetter('opinion_id'), valid_samples)}
        for metric in scores.keys():
            for opinion_id, agg in zip(data.keys(), scores[metric]):
                data[opinion_id][metric] = {k: v[0] for k, v in agg.items()}
        # data is shaped as
        # data = {123456: {'rouge-1': {'f': 0.1, 'r': 0.2, 'p': 0.3}, 'rouge-2': {'f': 0.4, 'r': 0.5, 'p': 0.6}},
        #         123457, {'rouge-1': {'f': xxx, 'r': xxx, 'p': xxx}, 'rouge-2': {'f': xxx, 'r': xxx, 'p': xxx}},

        frame = [{'opinion_id': opid, **{f'{metric.upper()}-{agg.upper()}': value
                                         for metric, l2d in l1d.items()
                                         for agg, value in l2d.items()}}
                 for opid, l1d in data.items()]
        # frame is shaped as
        # frame = [{'opinion_id': 123456, 'ROUGE-1-F': 0.1, 'ROUGE-2-R': 0.2, etc...},
        #          {'opinion_id': 123457, 'ROUGE-1-F': xxx, etc...}]

        if len(frame) > 0:
            file_name = os.path.join(_args.dest, f'{utils.random_name()}.parq')
            pd.DataFrame(frame).to_parquet(file_name)

        return len(x)

    os.makedirs(_args.dest, exist_ok=True)

    logging.info(f'Comparing summaries - Data from MongoDB {_args.mongodb} / {_args.db} / {_args.eval}')
    logging.info(f'Evaluation Metric: {_args.metric}')
    logging.info(f'To: {_args.dest}')

    workers = {
        'ngrams-cos': worker_evaluate_ngram_cosine,
        'rouge': worker_evaluate_rouge
    }

    with MongoClient(_args.mongodb) as client:
        @make_spin(Default, 'Querying MongoDB...')
        def _query() -> Tuple[Cursor, int]:
            collection = client[_args.db][_args.eval]
            _num_rows = collection.count_documents({})
            _cursor = collection.find({})
            return _cursor, _num_rows

        cursor, num_rows = _query()
        cursor_iterator_fn = utils.mongodb_cursor_iterator(cursor=cursor,
                                                           batch_size=_args.batch_size)
        utils.multiprocess(worker_fn=workers[_args.metric],
                           input_iterator_fn=cursor_iterator_fn,
                           total=num_rows,
                           nb_workers=_args.num_workers,
                           description='Evaluating Summaries')


def merge_verbatims():
    """
    Merge the verbatim dataset (made of tuple: citing_opinion / cited_opinion / verbatim) to a per-document summary
    dataset (made of tuple: opinion / summary), so we can later compare the summary with the verbatims.
    Run through the verbatim dataset and update all documents in the summary dataset with the verbatims by adding
    the verbatims one by one as a list to a field named 'verbatims'.

    :return:
    """
    @utils.queue_worker
    def worker_merge(x: List[Dict[str, Any]]) -> int:
        with MongoClient(_args.mongodb) as _client:
            for sample in x:
                if sample['score'] == -1:
                    continue

                cited_opinion_id = sample['cited_opinion_id']
                verbatim = sample['verbatim']
                logging.debug(f'Update {cited_opinion_id}')

                r = _client[_args.db][_args.summary].update_one(
                    filter={'opinion_id': cited_opinion_id},
                    update={'$push': {'verbatims': verbatim}}
                )

                if r.modified_count == 0:
                    logging.error(f'Could not update for opinion_id: {cited_opinion_id}')

        return len(x)

    with MongoClient(_args.mongodb) as client:
        @make_spin(Default, 'Querying MongoDB...')
        def _query() -> Tuple[Cursor, int]:
            collection = client[_args.db][_args.verbatim]
            _num_rows = collection.count_documents({})
            _cursor = collection.find({})
            return _cursor, _num_rows

        cursor, num_rows = _query()
        cursor_iterator_fn = utils.mongodb_cursor_iterator(cursor=cursor,
                                                           batch_size=_args.batch_size)
        utils.multiprocess(worker_fn=worker_merge,
                           input_iterator_fn=cursor_iterator_fn,
                           total=num_rows,
                           nb_workers=_args.num_workers,
                           description='Merging Summaries')


def parse_args(argstxt=None):
    if argstxt is None:
        argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    def default_parser(name: str,
                       description: str,
                       func: Callable[[None], None],
                       parallel: bool = True) -> ArgumentParser:
        # All parsers have a lot of common arguments
        new_parser = subparsers.add_parser(name=name, description=description)
        new_parser.add_argument('--mongodb', type=str, help='MongoDB Instance host:port')
        new_parser.add_argument('--db', type=str, help='Database name')
        if parallel:
            new_parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
            new_parser.add_argument('--batch-size', type=int, default=1024, help='Number of records sent at once to a '
                                                                                 'worker')
        new_parser.set_defaults(func=func)
        return new_parser

    # Merge the data, prepare for evaluation
    parser_merge = default_parser(
        name='merge',
        description='In preparation for evaluation, make a complete dataset where each document contains the summary'
                    'of an opinion and the collection of verbatims collected for this opinion.',
        func=merge_verbatims
    )
    parser_merge.add_argument('--verbatim', type=str, help='Collection name for the verbatims')
    parser_merge.add_argument('--summary', type=str, help='Collection name for the summaries')

    # Evaluate Summaries
    parser_evaluate = default_parser(
        name='summaries',
        description='Compare the summaries in dataset PATH to the summaries generated in dataset GROUND and record the'
                    'result in dataset DEST. Typically the GROUND dataset will be made of the verbatim quotes.',
        func=evaluate_summaries
    )
    parser_evaluate.add_argument('--dest', type=str, help='Destination Folder for the results')
    parser_evaluate.add_argument('--eval', type=str, help='Collection name of the summaries being evaluated')
    parser_evaluate.add_argument('--eval-field', type=str, help='Name of the field with the summaries in the evaluated'
                                                                 'collection')
    parser_evaluate.add_argument('--metric', type=str, choices=['rouge', 'ngrams-cos'])

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
