import glob
import os
import tarfile
import pandas as pd
import json
import sys
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import logging
import numpy as np
import requests
import datetime

from argparse import ArgumentParser, Namespace
from bs4 import BeautifulSoup
from pyspin.spin import make_spin, Default
from typing import Any
from tqdm import tqdm

from utils import multiprocess, queue_worker
from utils import parquet_dataset_iterator, file_list_iterator
from utils import random_name
from anchors import extract_anchors

# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args: Namespace = Namespace()

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


@queue_worker
def worker_extract_opinion(x: str) -> int:
    """
    Open a targz file, go through all contained JSON files, without extracting to disk, and in each file, extract
    a set of values, based on the list of tags given in args.tags. Builds a CSV file with all extracted data, one line
    per original JSON file.
    The CSV file is in the args.dest folder.
    Assumes the targz is made only of json files (it is the case in COURT LISTENER dataset)

    :param x: path to a tar.gz file containing multiple JSON files, one for each opinion
    :return: path to PARQUET file with the opinions
    """
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

    # We processed 1 file
    return 1


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

    targz_files = glob.glob(os.path.join(_args.targzpath, '*.tar.gz'))
    iterator, nb_files = file_list_iterator(targz_files)
    multiprocess(worker_fn=worker_extract_opinion,
                 input_iterator_fn=iterator,
                 total=nb_files,
                 nb_workers=_args.num_workers,
                 description='Extract opinion XML from archives')


def _transform_parquet_dataset(worker_fn):
    logging.info(f'FROM Dataset: {_args.path}')
    logging.info(f'TO Dataset: {_args.dest}')

    os.makedirs(_args.dest, exist_ok=True)

    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset  # we are sure of the actual class

    iterator, nb_rows = parquet_dataset_iterator(dataset)
    multiprocess(worker_fn=worker_fn,
                 input_iterator_fn=iterator,
                 total=nb_rows,
                 nb_workers=_args.num_workers,
                 description='Map citation network')


@queue_worker
def worker_extract_citation_map(x: pa.RecordBatch) -> int:
    """
    Get a list of 2-uple citing_opinion / cited_opinion.

    :param x: batch of opinions to parse
    :return: number of processed opinions
    """
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

    return x.num_rows


def create_citation_map():
    logging.info('Create CITATION MAP')
    _transform_parquet_dataset(worker_extract_citation_map)


@queue_worker
def worker_extract_anchor(x: pa.RecordBatch) -> int:
    """
    Extract for each citation, the ANCHOR of it, which is the catchphrase that introduces the citED opinion
    in the citING opinion. Each cited opinion is introduced by a sentence that underlines what from the cited
    opinion is an argument for the citing opinion.
    A parquet file is written on disk with the results.

    :param x: a batch (pyarrow.RecordBatch)
    :return: number of rows in the batch (int)
    """
    d = x.to_pydict()

    opinion_anchors = []
    for citing_opinion_id, opinion_html in zip(d['opinion_id'], d['html_with_citations']):
        op_anchors = extract_anchors(opinion_html, method=_args.method)
        opinion_anchors.extend([[citing_opinion_id, g['cited_opinion_id'], g['anchor']] for g in op_anchors])

    # Wrap-up, save to PARQUET file and return the number of processed rows
    df = pd.DataFrame(opinion_anchors, columns=['citing_opinion_id', 'cited_opinion_id', 'anchor'])
    for c in ['citing_opinion_id', 'cited_opinion_id']:
        df[c] = df[c].apply(pd.to_numeric)

    file_out = os.path.join(_args.dest, f'{random_name()}.parq')
    df.to_parquet(file_out)
    return x.num_rows


def create_anchor_dataset():
    logging.info(f'Create ANCHORS with method {_args.method}')
    _transform_parquet_dataset(worker_extract_anchor)


@queue_worker
def worker_summarize_textrank(x: pa.RecordBatch) -> int:
    """
    Applies textrank to summarize a collection of texts
    """
    # TODO
    return 0


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
        t = pa.Table.from_batches(batches=batches)        # Discard the warning 'parameter self unfilled'
        return t

    table = _collect()
    logging.info('Writing file')
    file_out = os.path.join(_args.dest, f'sample_{_args.num_samples}.parq')
    pq.write_table(table=table, where=file_out)


def download_courtlistener_opinions_bulk():
    url = "https://www.courtlistener.com/api/bulk-data/opinions/all.tar"

    block_size = 1024  # 1 Kibibyte
    now = datetime.date.today()
    filename = os.path.join(_args.to, f'{now.strftime("%Y%m%d")}_opinion.tar')

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))

    logging.info(f'Download Court Listener Opinion dataset, from URL {url}')
    logging.info(f'Destination file: {filename}')
    with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logging.error("ERROR, something went wrong")
        return

    if not _args.untar:
        return

    logging.info(f'Unpacking archive {filename}')
    with tarfile.open(filename, 'r') as tar:
        for m in tqdm(tar.getmembers()):
            tar.extract(m, path=_args.to)


def parse_args(argstxt=None):
    if argstxt is None:
        argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    # Download the Opinion dataset from CourtListener
    # Unpack it if requested
    parser_download = subparsers.add_parser('download')
    parser_download.add_argument('--to', type=str, help='Download folder')
    parser_download.add_argument('--untar', default=False, action='store_true', help='Untar the archive')
    parser_download.set_defaults(func=download_courtlistener_opinions_bulk)

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

    # Extract all ANCHORS from all citations
    parser_anchor = subparsers.add_parser('anchor')
    parser_anchor.add_argument('--method', type=str, choices=['nlp', 'last'], default='last',
                               help='Select the method to extract anchors.')
    parser_anchor.add_argument('--path', type=str, help='Path to the OPINION dataset folder')
    parser_anchor.add_argument('--dest', type=str, help='Destination folder for ANCHOR dataset')
    parser_anchor.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser_anchor.set_defaults(func=create_anchor_dataset)

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
