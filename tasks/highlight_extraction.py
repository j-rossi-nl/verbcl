import glob
import json
import logging
import os
import pandas as pd
import re
import shutil
import sys

from argparse import ArgumentParser, Namespace
from nltk.tokenize import word_tokenize
from typing import Any, List

from bert_classifier import HighlightSentenceClassifier, PandasToDataset
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
            "origin": "test_only_presumm.parq",
            "dest": "cl.test.{}.json"
        }
    }

    only = {k: v for k, v in splits.items() if k in _args.only}

    for k, v in only.items():
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
        # highlight has the binary indicator whether this sentence was a highlight or not
        data = [(opinion_id, d.sort_values('sentence_id')[['raw_text', 'highlight']])
                for opinion_id, d in data_df.groupby('opinion_id')]

        iterator, nb_opinions = batch_iterator(items=data, batch_size=_args.batch_size)
        multiprocess(worker_fn=_batch_to_jsonl, input_iterator_fn=iterator, total=nb_opinions,
                     nb_workers=_args.num_workers, description='Transforming to JSON')


def build_bert_dataset():
    pd2ds = PandasToDataset(_args.save, _args.batch_size, _args.num_workers)
    logging.info(f"Reading data from {_args.data}")
    df = pd.read_parquet(_args.data)

    if _args.limit_presumm is not None:
        with open(_args.limit_presumm) as src:
            presumm_results = [json.loads(line) for line in src]
        presumm = {x["opinion_id"]: len(x["gold_ids"]) for x in presumm_results if sum(x["gold_ids"]) > 0}
        df = df[df["opinion_id"].isin(presumm.keys())]
        df = df[df[["opinion_id", "sentence_id"]].apply(lambda x: x["sentence_id"] < presumm[x["opinion_id"]], axis=1)]
        logging.info("Filtering to match PreSumm limitations.")
        logging.info(f"Data size: {df.shape[0]} samples. With {len(df['opinion_id'].unique())} OpinionIds")

    logging.info(f"Storing in {_args.save}")
    pd2ds(df)


def run_textrank():
    tmpfolder = os.path.join("/tmp", f"juju_{random_name()}/")
    os.makedirs(tmpfolder)

    # If we decide to align with PreSumm, we run textrank on only the first N sentences
    # We get the number of sentences per opinion by having a look at the predictions returned by PreSumm
    # When there is at least a positive sentence in the gold_ids labels
    presumm = {}
    if _args.limit_presumm is not None:
        with open(_args.limit_presumm) as src:
            presumm_results = [json.loads(line) for line in src]
        presumm = {x["opinion_id"]: len(x["gold_ids"]) for x in presumm_results if sum(x["gold_ids"]) > 0}

    @queue_worker
    def _textrank(list_of_df: List[Any]) -> int:
        nlp = CustomNLP()
        with open(os.path.join(tmpfolder, f"{random_name(16)}.run"), "w") as tmp_out:
            for opinion_id, df in list_of_df:
                try:
                    df: pd.DataFrame
                    gold = df['highlight'].astype(int).values.tolist()

                    df['raw_text'] = df['raw_text'].str.replace(newline_pattern, " ", regex=True)

                    # Mimic PreSumm = limit to a number of sentences
                    # If the opinion is not part of those where there is at least a positive sentence in the
                    # first N, then we skip the opinion.
                    num_phrases = len(df)
                    if _args.limit_presumm is not None:
                        if opinion_id not in presumm:
                            continue
                        num_phrases = presumm[opinion_id]

                    doc = nlp(df['raw_text'][:num_phrases].values.tolist())
                    # noinspection PyProtectedMember
                    tr = doc._.textrank
                    preds = [1 - s.distance for s in tr.calc_sent_dist(_args.limit_phrases)]

                    for sentence_id, score in enumerate(preds):
                        tmp_out.write(f"{opinion_id}\tQ0\ts{sentence_id}\t0\t{score}\tTEXTRANK\n")
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
        for f in glob.glob(os.path.join(tmpfolder, "*.run")):
            with open(f) as src:
                out.write(src.read())

    # Clean TMP folder
    shutil.rmtree(tmpfolder)


def train_bert():
    logging.info("Training BERT-like model for sentence classification.")
    logging.info("Creating Model")
    clf = HighlightSentenceClassifier(save_folder=_args.save)

    logging.info(f"Reading Training data from {_args.train_data}")
    logging.info("Starting Training")
    clf.train(_args.train_data)


def test_bert():
    logging.info("Prediction BERT-like model for sentence classification.")
    logging.info("Creating Model")
    clf = HighlightSentenceClassifier(save_folder=_args.model, load_checkpoint=True)
    logging.info("Starting Prediction")
    clf.test(_args.test_data, _args.save)


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
    parser_presumm.add_argument('--only', nargs='*', default=['train', 'test'], help='Which datasets splits to prepare')
    parser_presumm.add_argument('--num-workers', type=int, default=4, help='Parallel workers.')
    parser_presumm.add_argument('--batch-size', type=int, default=32, help='Batch size for parallel workers')
    parser_presumm.set_defaults(func=build_presumm_dataset)

    parser_textrank = subparsers.add_parser(name='run-textrank', description='Run textrank on the dataset, '
                                                                             'Produce a report')
    parser_textrank.add_argument('--path', type=str, help='Dataset. (test.parq)')
    parser_textrank.add_argument('--dest', type=str, help='Report JSONL path')
    parser_textrank.add_argument('--limit-presumm', type=str, help='Provide the JSON file output of PreSumm inference'
                                                                   'on test dataset. TextRank will run only on the'
                                                                   'opinions where there is 1 highlight in the part'
                                                                   'of the text that PreSumm worked on.')
    parser_textrank.add_argument('--limit-phrases', type=int, default=10, help='limit_phrases from pytextrank')
    parser_textrank.add_argument('--num-workers', type=int, default=4, help='Parallel workers.')
    parser_textrank.add_argument('--batch-size', type=int, default=32, help='Batch size for parallel workers')
    parser_textrank.set_defaults(func=run_textrank)

    parser_build_bert = subparsers.add_parser(name="build-bert", description="Build the data for BERT")
    parser_build_bert.add_argument('--data', type=str, help="Dataset xxx.parq")
    parser_build_bert.add_argument('--limit-presumm', type=str,
                                   help='Provide the JSON file output of PreSumm inference on test dataset. BERT will '
                                        'run only on the opinions where there is 1 highlight in the part of the text '
                                        'that PreSumm worked on.')
    parser_build_bert.add_argument('--batch-size', type=int, help="Batch size")
    parser_build_bert.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser_build_bert.add_argument('--save', type=str, help='Folder where the data will be stored')
    parser_build_bert.set_defaults(func=build_bert_dataset)

    parser_train_bert = subparsers.add_parser(name="train-bert", description="Train a BERT model")
    parser_train_bert.add_argument('--train-data', type=str, help='Dataset (train.parq)')
    parser_train_bert.add_argument('--save', type=str, help='Folder where models are saved')
    parser_train_bert.set_defaults(func=train_bert)

    parser_testbert = subparsers.add_parser(name="test-bert", description="Use BERT Model for inference")
    parser_testbert.add_argument('--model', type=str, help='Folder with checkpoints')
    parser_testbert.add_argument('--test-data', type=str, help='Dataset (train.parq)')
    parser_testbert.add_argument('--save', type=str, help='Save the inference results')
    parser_testbert.set_defaults(func=test_bert)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
