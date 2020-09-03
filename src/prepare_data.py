import glob
import os
import tarfile
import pandas as pd
import json
import sys
import en_core_web_sm
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import logging
import numpy as np

from argparse import ArgumentParser, Namespace
from multiprocessing import Queue
from bs4 import BeautifulSoup
from pyspin.spin import make_spin, Default

from utils import multiprocess_with_queue, multiprocess_dataset, random_name
from gist_extraction import extract_catchphrase

# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args: Namespace = Namespace()

# The SpaCy NLP engine
_nlp = en_core_web_sm.load()

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def worker_extract_opinion(in_queue: Queue, out_queue: Queue) -> None:
    """
    Open a targz file, go through all contained JSON files, without extracting to disk, and in each file, extract
    a set of values, based on the list of tags given in args.tags. Builds a CSV file with all extracted data, one line
    per original JSON file.
    The CSV file is in the args.dest folder.
    Assumes the targz is made only of json files (it is the case in COURT LISTENER dataset)

    :param in_queue: queue
    :param out_queue: queue
    :return:
    """
    while True:
        x: str = in_queue.get(True)

        data = []
        try:
            with tarfile.open(x, mode='r:*') as tgz:
                json_list = tgz.getmembers()
                for json_member in json_list:
                    # Extract to memory each JSON file
                    json_file = tgz.extractfile(json_member)
                    js = json.load(json_file)
                    if not all([t in js for t in _args.tags]):
                        continue
                    data.append([js[t] for t in _args.tags])
        except Exception as caught:
            print('Processing {}, Exception {}'.format(x, caught))

        file_out = os.path.join(_args.dest, f'{os.path.basename(x)[:-7]}.parq')
        df = pd.DataFrame(data, columns=_args.tags)

        # The opinion id is the field 'id'. For clarity, rename it 'opinion_id'
        if 'id' in df.columns:
            df['opinion_id'] = df['id'].apply(pd.to_numeric)
            df = df.drop(columns=['id'])

        # Some courts just don't have data
        if df.shape[0] > 0:
            df.to_parquet(path=file_out)
        else:
            # None signals to the worker that nothing was produced
            file_out = None

        out_queue.put(file_out)


def create_opinion_dataset():
    """
    Extract values from JSON files in targz files. Each targz file will generate a PARQUET file.

    :return:
    """
    logging.info('Converting from CourtListener format to Parquet Dataset')
    logging.info(f'Extracting Tags: {" / ".join(_args.tags)}')
    logging.info(f'From folder: {_args.targzpath}')
    logging.info(f'To folder: {_args.dest}')
    os.makedirs(_args.dest, exist_ok=True)

    def _targz_iterator():
        for p in glob.glob(os.path.join(_args.targzpath, '*.tar.gz')):
            yield p

    _ = multiprocess_with_queue(worker_fn=worker_extract_opinion,
                                input_iterator_fn=_targz_iterator,
                                nb_workers=_args.num_workers,
                                description='Extract opinion XML from archives')


def worker_extract_citation_map(in_queue: Queue, out_queue: Queue) -> None:
    """
    Get a list of 2-uple citing_opinion / cited_opinion.

    :param in_queue: tasks to do
    :param out_queue: done tasks
    :return:
    """

    while True:
        x: pa.RecordBatch = in_queue.get(True)
        d = x.to_pydict()

        citations = []
        for citing_opinion_id, opinion_html in zip(d['opinion_id'], d['html_with_citations']):
            citing_opinion_id = citing_opinion_id
            try:
                bs = BeautifulSoup(opinion_html, 'html.parser')
            except TypeError:
                continue
            search = {'class': 'citation', 'data-id': True}
            cited_opinions = set([int(x['data-id']) for x in bs.find_all('span', **search)])
            citations.extend([[citing_opinion_id, x] for x in cited_opinions])

        df = pd.DataFrame(citations, columns=['citing_opinion_id', 'cited_opinion_id'])
        for c in ['citing_opinion_id', 'cited_opinion_id']:
            df[c] = df[c].apply(pd.to_numeric)

        if len(df) > 0:
            file_out = os.path.join(_args.dest, f'{random_name()}.parq')
            df.to_parquet(file_out)
        out_queue.put(x.num_rows)


def create_citation_map():
    os.makedirs(_args.dest, exist_ok=True)
    multiprocess_dataset(worker_fn=worker_extract_citation_map,
                         input_dataset_path=_args.path,
                         nb_workers=_args.num_workers,
                         description='Map citation network')


def worker_extract_gist(in_queue: Queue, out_queue: Queue) -> None:
    """
    Extract for each citation, the 'GIST' of it, which is the catchphrase that introduces the citED opinion
    in the citING opinion. Each cited opinion is introduced by a sentence that underlines what from the cited
    opinion is an argument for the citing opinion.

    :param in_queue: tasks to do
    :param out_queue: done tasks
    :return: the path to a CSV file with columns citing_opinion_id, cited_opinion_id, extracted_gist
    """

    while True:
        x: pa.RecordBatch = in_queue.get(True)
        d = x.to_pydict()

        opinion_gists = []
        for citing_opinion_id, opinion_html in zip(d['opinion_id'], d['html_with_citations']):
            op_gists = extract_catchphrase(opinion_html, method=_args.method)
            opinion_gists.extend([[citing_opinion_id, g['cited_opinion_id'], g['gist']] for g in op_gists])

        # Wrap-up, save to CSV file and return the name of that CSV file to caller
        df = pd.DataFrame(opinion_gists, columns=['citing_opinion_id', 'cited_opinion_id', 'gist'])
        for c in ['citing_opinion_id', 'cited_opinion_id']:
            df[c] = df[c].apply(pd.to_numeric)

        file_out = os.path.join(_args.dest, f'{random_name()}.parq')
        df.to_parquet(file_out)
        out_queue.put(x.num_rows)


def create_gist_dataset():
    multiprocess_dataset(worker_fn=worker_extract_gist,
                         input_dataset_path=_args.path,
                         nb_workers=_args.num_workers,
                         description='Gist')


def worker_summarize_textrank(in_queue: Queue, out_queue: Queue) -> None:
    """
    Applies textrank to summarize a collection of texts
    """
    # TODO
    pass


def create_opinion_summary_dataset():
    """
    Uses all texts from the CSV file args.texts (column args.tag), summarize acording to args.method and outputs a
    CSV file with fields 'opinion_id' and 'summary'
    """
    # TODO
    pass


def create_sample_dataset():
    """
    Extract a random sample from the OPINION dataset.

    :return:
    """
    logging.info('Extract a random sample of opinions')
    logging.info(f'Opinion Dataset: {_args.path}')
    logging.info(f'Extract to: {_args.dest}')
    logging.info(f'Nb samples: {_args.num_samples}')

    os.makedirs(_args.dest, exist_ok=True)

    logging.info('Load citation map...')
    citation_map = pd.read_csv(_args.citation_map)
    dataset = ds.dataset(_args.path)

    logging.info('Random sample...')
    pool_opinion_ids = citation_map['citing_opinion_id'].unique()
    sample_opinion_ids = np.random.choice(a=pool_opinion_ids, size=_args.num_samples, replace=False)

    @make_spin(Default, "Collecting opinions... (might take a few minutes)")
    def _collect():
        scan_tasks = dataset.scan(filter=ds.field('opinion_id').isin(sample_opinion_ids))
        batches = sum((list(s.execute()) for s in scan_tasks), [])
        table = pa.Table.from_batches(batches=batches)
        return table

    table = _collect()
    logging.info('Writing file')
    file_out = os.path.join(_args.dest, f'sample_{_args.num_samples}.parq')
    pq.write_table(table=table, where=file_out)


def parse_args(argstxt=None):
    if argstxt is None:
        argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    # From CourtListener to Parquet dataset
    parser_tocsv = subparsers.add_parser('opinion')
    parser_tocsv.add_argument("--targzpath", type=str, help="The path to the folder with all *.tar.gz "
                                                            "files to be decompressed")
    parser_tocsv.add_argument('--dest', type=str, help='Destination folder for the PARQUET files')
    parser_tocsv.add_argument('--tags', nargs='+', help='List of tags to get for each opinion JSON file '
                                                        'in each tar,gz file in folder')
    parser_tocsv.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser_tocsv.set_defaults(func=create_opinion_dataset)

    # Random sample from the opinion dataset
    parser_sample = subparsers.add_parser('sample')
    parser_sample.add_argument('--path', type=str, help='Path to the OPINION dataset folder')
    parser_sample.add_argument('--dest', type=str, help='Destination folder for SAMPLE dataset')
    parser_sample.add_argument('--citation-map', type=str, help='CSV File of the citation map')
    parser_sample.add_argument('--num-samples', type=int, default=10000, help='Number of samples')
    parser_sample.set_defaults(func=create_sample_dataset)

    # Extract all GIST from all citations
    parser_gist = subparsers.add_parser('gist')
    parser_gist.add_argument('--method', type=str, choices=['nlp', 'last'], default='last',
                             help='Select the method to extract gist.')
    parser_gist.add_argument('--path', type=str, help='Path to the OPINION dataset folder')
    parser_gist.add_argument('--dest', type=str, help='Destination folder for GIST dataset')
    parser_gist.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser_gist.set_defaults(func=create_gist_dataset)

    # Create a CSV file to track all couples of (citing, cited) opinions
    parser_citations = subparsers.add_parser('citation-map')
    parser_citations.add_argument('--path', type=str, help='Path to the Opinion PARQUET dataset')
    parser_citations.add_argument('--dest', type=str, help='Destination folder for PARQUET files')
    parser_citations.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser_citations.set_defaults(func=create_citation_map)

    # Summarize each opinion
    parser_opsum = subparsers.add_parser('opinion-summary')
    parser_opsum.add_argument('--path', type=str, help='Path to the OPINION Parquet dataset')
    parser_opsum.add_argument('--dest', type=str, help='Path to the OPINION SUMMARY Parquet dataset folder')
    parser_opsum.add_argument('--method', type=str, choices=['textrank'], help='Summarization technic')
    parser_opsum.add_argument('--num-words', type=int, default=200, help='Target length for the summary, '
                                                                         'in number of words')
    parser_opsum.add_argument('--num-workers', type=int, help='Number of parallel workers')
    parser_opsum.set_defaults(func=create_opinion_summary_dataset)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
#    try:
#    except Exception as excp:
#        import pdb
#        pdb.post_mortem()
