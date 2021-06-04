import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import pytrec_eval
import re
import shutil
import sys

from argparse import ArgumentParser, Namespace
from nltk.tokenize import word_tokenize
from typing import Any, List

from utils import batch_iterator, config, gather_by_opinion_ids, make_clean_folder, multiprocess, \
    queue_worker, random_name
from utils import CustomNLP, OpinionSentence


config()
_args = Namespace()

newline_pattern = re.compile(r"[\n\t]")


def build_datasets():
    path = _args.path
    files = ["train.csv", "test.csv"]
    for f in files:
        file_path = os.path.join(path, f)
        assert os.path.isfile(file_path)

        logging.info(f"Processing {f}")

        with open(file_path) as csv:
            ids: List[int] = list(map(lambda x: int(x.strip()), csv.readlines()))

        df = gather_by_opinion_ids(class_=OpinionSentence, opinion_ids=ids, envfile=_args.env,
                                   nb_workers=_args.num_workers, batch_size=_args.batch_size)
        save_path = os.path.join(path, f"{os.path.splitext(f)[0]}.parq")
        df.to_parquet(save_path)


def build_presumm_dataset():
    make_clean_folder(_args.dest)
    splits = {
        "train": {
            "origin": "train.parq",
            "dest": "cl.train.{}.json"
        },
        "test": {
            "origin": "test.parq",
            "dest": "cl.test.{}.json"
        }
    }

    for k, v in splits.items():
        logging.info(f"Now Processing {k}. Reading file {v['origin']}")

        @queue_worker
        def _batch_to_jsonl(list_of_df: List[Any]) -> int:
            d = []
            for opinion_id, df in list_of_df:
                df: pd.DataFrame
                df['raw_text'] = df['raw_text'].str.replace(newline_pattern, " ", regex=True)
                src = [word_tokenize(x) for x in df['raw_text']]
                tgt = [word_tokenize(x) for x in df[df['highlight'] == 1]['raw_text']]
                d.append({"src": src, "tgt": tgt, "opinion_id": opinion_id})

            dest_file = os.path.join(_args.dest, v["dest"].format(random_name()))
            with open(dest_file, 'w') as out:
                json.dump(d, out)
            return len(list_of_df)

        data_df: pd.DataFrame = pd.read_parquet(os.path.join(_args.path, v["origin"]))

        # data is a list of dataframes. Each dataframes has 2 columns: raw_text has the text of the sentences,
        # highlight has th ebinary indicator whether this sentence was a highlight or not
        data = [(opinion_id, d.sort_values('sentence_id')[['raw_text', 'highlight']])
                for opinion_id, d in data_df.groupby('opinion_id')]

        iterator, nb_opinions = batch_iterator(items=data, batch_size=_args.batch_size)
        multiprocess(worker_fn=_batch_to_jsonl, input_iterator_fn=iterator, total=nb_opinions,
                     nb_workers=_args.num_workers, description='Transforming to JSON')


def evaluate():
    logging.info(f"Loading QRELS from {_args.qrels}")
    with open(_args.qrels) as src:
        qrels = json.load(src)

    logging.info(f"Loading RUN from {_args.run}")
    with open(_args.run) as src:
        data = [json.loads(line) for line in src]
    logging.info(f"Number of Samples in RUN: {len(data)}")

    # Align run and QREL. We have opinion_id that are in the run but not in the QRELS
    select_opinion_ids = list(map(int, qrels.keys()))
    good_data = list(filter(lambda x: x["opinion_id"] in select_opinion_ids, data))
    logging.info(f"Keeping {len(good_data)} Samples from RUN")

    # Each JSON contains 2 fields
    # opinion_id
    # pred_ids: vector of length N, preds_ids[i] has score of sentence i
    # Score Adaptation: TREC expects scores on the rule "HIGHER = BETTER"
    if _args.method == "presumm":
        # PRESUMM gives list of sentence ids ranked by their score
        # Each sentence should get as a score its reverse position in the ranked list
        run = {
            str(d["opinion_id"]): {
                f"s{sentence_id}": 1 + len(d["pred_ids"]) - rank
                for rank, sentence_id in enumerate(d["pred_ids"])
            }
            for d in good_data
        }
    else:
        # No adaptation needed
        run = {
            str(d["opinion_id"]): {
                f"s{i}": score for i, score in enumerate(d["pred_ids"])
            }
            for d in good_data
        }

    # Define the metrics
    parameterized_measures = {
        "recall": list(range(1, 6)),
        "P": list(range(1, 6)),
    }
    standard_measures = {
        "recip_rank"
    }
    measures = [f"{m}.{p}" for m, params in parameterized_measures.items() for p in params] + list(standard_measures)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    results = evaluator.evaluate(run)

    logging.info(f"Store results in {_args.dest}")
    with open(_args.dest, "w") as out:
        json.dump(results, out)

    aggregates = {
        measure: np.mean([x[measure] for x in results.values()])
        for measure in map(lambda x: x.replace('.', '_'), measures)
    }

    print('\n'.join(f'{k:<15}: {v:>0.2f}' for k, v in aggregates.items()))

    logging.info(f"Store Aggregated results in {_args.agg}")
    with open(_args.agg, "w") as out:
        json.dump(aggregates, out)


def run_textrank():
    tmpfolder = os.path.join("/tmp", f"juju_{random_name()}/")
    os.makedirs(tmpfolder)

    @queue_worker
    def _textrank(list_of_df: List[Any]) -> int:
        nlp = CustomNLP()
        with open(os.path.join(tmpfolder, f"{random_name(16)}.json"), "w") as tmp_out:
            for opinion_id, df in list_of_df:
                try:
                    df: pd.DataFrame
                    gold = df['highlight'].astype(int).values.tolist()

                    df['raw_text'] = df['raw_text'].str.replace(newline_pattern, " ", regex=True)
                    doc = nlp(df['raw_text'].values.tolist())
                    # noinspection PyProtectedMember
                    tr = doc._.textrank
                    preds = [1 - s.distance for s in tr.calc_sent_dist(_args.limit_phrases)]

                    d = {
                        "opinion_id": opinion_id,
                        "gold_ids": gold,
                        "pred_ids": preds
                    }

                    tmp_out.write(json.dumps(d) + '\n')
                except Exception as e:
                    logging.error(f"Could not process Opinion {opinion_id}: {repr(e)}")
                    continue

        return len(list_of_df)

    logging.info(f"Loading {_args.path}")
    data_df: pd.DataFrame = pd.read_parquet(_args.path)

    # data is a list of dataframes. Each dataframes has 2 columns: raw_text has the text of the sentences,
    # highlight has th ebinary indicator whether this sentence was a highlight or not
    data = [(opinion_id, d.sort_values('sentence_id')[['raw_text', 'highlight']]) for opinion_id, d in
            data_df.groupby('opinion_id')]
    iterator, nb_opinions = batch_iterator(items=data, batch_size=_args.batch_size)
    multiprocess(worker_fn=_textrank, input_iterator_fn=iterator, total=nb_opinions,
                 nb_workers=_args.num_workers, description=f'Summarize with TextRank.')

    # Gather all data
    with open(_args.dest, "w") as out:
        for f in glob.glob(os.path.join(tmpfolder, "*.json")):
            with open(f) as src:
                out.write(src.read())

    # Clean TMP folder
    shutil.rmtree(tmpfolder)


def parse_args():
    argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    parser_build = subparsers.add_parser(name='build-dataset', description='Create the PARQUET TRAIN and TEST dataset')
    parser_build.add_argument('--path', type=str, help='Path to the folder with train.csv and test.csv.')
    parser_build.add_argument('--env', type=str, help='Path to the env file for Elastic connection')
    parser_build.add_argument('--num-workers', type=int, default=4, help='Parallel workers.')
    parser_build.add_argument('--batch-size', type=int, default=32, help='Batch size for parallel workers')
    parser_build.set_defaults(func=build_datasets)

    parser_presumm = subparsers.add_parser(name='build-presumm', description='Build the dataset for training PRESUMM')
    parser_presumm.add_argument('--path', type=str, help='Path to the folder with the current data')
    parser_presumm.add_argument('--dest', type=str, help='Path to the folder with the PRESUMM data')
    parser_presumm.add_argument('--num-workers', type=int, default=4, help='Parallel workers.')
    parser_presumm.add_argument('--batch-size', type=int, default=32, help='Batch size for parallel workers')
    parser_presumm.set_defaults(func=build_presumm_dataset)

    parser_textrank = subparsers.add_parser(name='run-textrank', description='Run textrank on the dataset, '
                                                                             'Produce a report')
    parser_textrank.add_argument('--path', type=str, help='Dataset. (test.parq)')
    parser_textrank.add_argument('--dest', type=str, help='Report JSONL path')
    parser_textrank.add_argument('--limit-phrases', type=int, default=10, help='limit_phrases from pytextrank')
    parser_textrank.add_argument('--num-workers', type=int, default=4, help='Parallel workers.')
    parser_textrank.add_argument('--batch-size', type=int, default=32, help='Batch size for parallel workers')
    parser_textrank.set_defaults(func=run_textrank)

    parser_evaluate = subparsers.add_parser(name='evaluate', description='Compute the evaluation statistics')
    parser_evaluate.add_argument('--qrels', type=str, help='Path to QRELS JSON')
    parser_evaluate.add_argument('--run', type=str, help='Path to report JSONL')
    parser_evaluate.add_argument('--method', type=str, choices=['presumm', 'textrank'], help='Which model did generate '
                                                                                             'the report')
    parser_evaluate.add_argument('--dest', type=str, default='result.json', help='JSON file for storing results')
    parser_evaluate.add_argument('--agg', type=str, default='agg_result.json', help='JSON file for storing '
                                                                                    'aggregated results')
    parser_evaluate.set_defaults(func=evaluate)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()

    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
