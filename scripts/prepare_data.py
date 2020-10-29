import datetime
import glob
import json
import logging
import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests
import tarfile
import shutil
import sys

import utils

from argparse import ArgumentParser, Namespace
from nltk.tokenize import sent_tokenize
from pyspin.spin import make_spin, Default
from typing import Any, Callable, Optional
from tqdm import tqdm

from courtlistener import Opinion, OpinionDataset, opinions_in_arrowbatch

# Configure
utils.config()


# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args: Namespace = Namespace()


def _transform_parquet_dataset(worker_fn):
    logging.info(f'FROM Dataset: {_args.path}')
    logging.info(f'TO Dataset: {_args.dest}')

    os.makedirs(_args.dest, exist_ok=True)

    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset  # we are sure of the actual class

    iterator, nb_rows = utils.parquet_dataset_iterator(dataset, batch_size=_args.batch_size)
    utils.multiprocess(worker_fn=worker_fn,
                       input_iterator_fn=iterator,
                       total=nb_rows,
                       nb_workers=_args.num_workers,
                       description='Progress')


def create_annotation_dataset():
    """
    Uses an opinion dataset and create the text snippets that will be manually annotated.
    The output is a collection of TXT files.
    """
    @utils.queue_worker
    def worker_jsonl_for_annotation(x: pa.RecordBatch) -> int:
        """
        Extract the text to be annotated for anchor extraction.
        """
        file_out = os.path.join(_args.dest, f'{utils.random_name()}.json')
        with open(file_out, 'w', encoding='utf-8') as out:
            out.write('\n'.join(j for op in opinions_in_arrowbatch(x)
                                for j in op.doccano(max_words_before_after=_args.max_words_extract)))

        return x.num_rows

    logging.info(f'Create JSONL for ANNOTATIONS')
    _transform_parquet_dataset(worker_jsonl_for_annotation)


def create_citation_map():
    @utils.queue_worker
    def worker_extract_citation_map(x: pa.RecordBatch) -> int:
        """
        Get a list of 2-uple citing_opinion / cited_opinion.

        :param x: batch of opinions to parse
        :return: number of processed opinions
        """
        citations = []
        for opinion in opinions_in_arrowbatch(x):
            citing_opinion_id = opinion.opinion_id
            cited_opinions = set(d['cited_opinion_id'] for d in opinion.citations())
            citations.extend([{'citing_opinion_id': citing_opinion_id, 'cited_opinion_id': x} for x in cited_opinions])

        df = pd.DataFrame(citations)
        for c in df.columns:
            df[c] = df[c].apply(pd.to_numeric)

        if len(df) > 0:
            file_out = os.path.join(_args.dest, f'{utils.random_name()}.parq')
            df.to_parquet(file_out)

        return x.num_rows

    logging.info('Create CITATION MAP')
    _transform_parquet_dataset(worker_extract_citation_map)


def create_opinion_dataset():
    """
    Extract values from JSON files in targz files. Each targz file will generate a PARQUET file.

    :return:
    """
    @utils.queue_worker
    def worker_extract_opinion(x: str) -> int:
        """
        Open a targz file, go through all contained JSON files, without extracting to disk, and in each file, extract
        a set of values, based on the list of tags given in args.tags. Builds a CSV file with all extracted data,
        one line per original JSON file.
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

    logging.info('Converting from CourtListener format to Parquet Dataset')
    logging.info(f'Extracting Tags: {" / ".join(_args.tags)}')
    logging.info(f'From folder: {_args.path}')
    logging.info(f'To folder: {_args.dest}')
    os.makedirs(_args.dest, exist_ok=True)

    targz_files = glob.glob(os.path.join(_args.path, '*.tar.gz'))
    iterator, nb_files = utils.file_list_iterator(targz_files)
    utils.multiprocess(worker_fn=worker_extract_opinion,
                       input_iterator_fn=iterator,
                       total=nb_files,
                       nb_workers=_args.num_workers,
                       description='Extract opinion XML from archives')


def create_prophetnet():
    """
    Create original data ready for preprocessing and summarization with ProphetNet.
    For each opinion in the dataset, the full text is written as one line with sentences separated by `<S_SEP>`.
    This file is ready to be preprocessed like cnndm from prophetnet, etc...
    Refer to https://github.com/microsoft/ProphetNet for further instructions
    :return:
    """
    @utils.queue_worker
    def worker_prepare_prophetnet(x: pa.RecordBatch) -> int:
        texts = []
        ids = []
        for opinion in opinions_in_arrowbatch(x):
            full_text = '<S_SEP>'.join(sent_tokenize(opinion.raw_text)).replace('\n', '').replace('\r', '')
            texts.append(full_text)
            ids.append(str(opinion.opinion_id))

        fout = os.path.join(_args.dest, utils.random_name())
        fout_txt = fout + '.txt'
        fout_idx = fout + '.idx'

        with open(fout_txt, 'w') as f:
            f.write('\n'.join(texts))

        with open(fout_idx, 'w') as f:
            f.write('\n'.join(ids))

        return x.num_rows

    logging.info('Creating ProphetNet input dataset')
    _transform_parquet_dataset(worker_prepare_prophetnet)

    # Gather all txt files into 1 and prepare the index
    txts = glob.glob(os.path.join(_args.dest, '*.txt'))
    idxs = [x[:-4]+'.idx' for x in txts]

    opinions_file = os.path.join(_args.dest, 'opinions.txt')
    index_file = os.path.join(_args.dest, 'opinions.idx')
    with open(opinions_file, 'w') as all_texts:
        with open(index_file, 'w') as all_idxs:
            for txt, idx in zip(txts, idxs):
                all_texts.write(open(txt).read())
                all_idxs.write(open(idx).read())

                os.remove(txt)
                os.remove(idx)

    if _args.export is not None:
        if os.path.isdir(_args.export):
            shutil.copy(opinions_file, _args.export)
            shutil.copy(index_file, _args.export)
        else:
            logging.error(f'No export to ProphetNet as {_args.export} is not a folder.')
            return


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

    def _collect_opinions(opinion_ids: np.ndarray,
                          file_out: str):
        @make_spin(Default, "Collecting opinions... (might take a few minutes)")
        def _do_collect():
            scan_tasks = dataset.scan(filter=ds.field('opinion_id').isin(opinion_ids))
            batches = sum((list(s.execute()) for s in scan_tasks), [])
            # noinspection PyArgumentList
            t = pa.Table.from_batches(batches=batches)  # incorrect warning about this call
            return t

        table = _do_collect()
        pq.write_table(table=table, where=file_out)

    def _dest_file(name: str) -> str:
        return os.path.join(_args.dest, name)

    logging.info('Load citation map...')
    citation_path = _args.citation_map
    assert os.path.isdir(citation_path) or os.path.isfile(citation_path)
    if os.path.isdir(citation_path):
        citation_map = ds.dataset(citation_path).to_table().to_pandas()
    else:
        citation_map = pd.read_csv(_args.citation_map)
    dataset = ds.dataset(_args.path)

    logging.info('Random sample...')
    pool_opinion_ids = citation_map['citing_opinion_id'].unique()

    todo = []
    sample_opinion_ids: np.ndarray = np.random.choice(a=pool_opinion_ids, size=_args.num_samples, replace=False)
    todo.append({'ids': sample_opinion_ids, 'filename': 'sample_core.parq'})

    if _args.add_cited:
        cited_opinion_ids = citation_map[citation_map['citing_opinion_id'].isin(sample_opinion_ids)]['cited_opinion_id']
        todo.append({'ids': cited_opinion_ids, 'filename': 'sample_cited.parq'})

    if _args.add_citing:
        citing_opinion_ids = citation_map[citation_map['cited_opinion_id'].isin(sample_opinion_ids)][
            'citing_opinion_id']
        todo.append({'ids': citing_opinion_ids, 'filename': 'sample_citing.parq'})

    _ = list(map(lambda x: _collect_opinions(x['ids'], _dest_file(x['filename'])), todo))


def create_summary_dataset():
    @utils.queue_worker
    def worker_summary(x: pa.RecordBatch) -> int:
        """
        Create a summary of opinions.

        :param x: a batch of opinions
        :return: number of processed records
        """
        df = pd.DataFrame([{'opinion_id': opinion.opinion_id,
                            f'summary_{_args.method}': utils.summarization_methods[_args.method](opinion.raw_text)}
                           for opinion in opinions_in_arrowbatch(x)])
        file_out = os.path.join(_args.dest, f'{utils.random_name()}.parq')
        df.to_parquet(file_out)
        return x.num_rows

    logging.info(f'Create SUMMARIES of utils.Opinions')
    _transform_parquet_dataset(worker_summary)


def verify_sample():
    """
    Filter the CORE sample to keep only those whose citations are all within the CITED dataset. Overwrites the
    PARQUET file containing the CORE sample.

    :return:
    """
    logging.info('Verify an extracted sample of opinions')
    logging.info(f'CORE dataset: {_args.core}')
    logging.info(f'CITED dataset: {_args.cited}')

    core = pd.read_parquet(_args.core)
    cited_ids = pd.read_parquet(_args.cited)['opinion_id'].unique()
    citing_missing_cited_ids = [
        data['opinion_id'] for _, data in tqdm(core.iterrows(), total=core.shape[0])
        if any(c['cited_opinion_id'] not in cited_ids for c in Opinion(data['opinion_id'],
                                                                       data['html_with_citations']).citations())]

    if len(citing_missing_cited_ids) == 0:
        logging.info('Dataset is clean')
        logging.info('All opinions cited in CORE were located in CITED')
        return

    logging.warning('Dataset is not clean. There are missing references in CITED')
    for op_id in citing_missing_cited_ids:
        logging.warning(f'CORE id {op_id} cites opinion(s) that are not in CITED dataset')

    if _args.clean:
        logging.warning('Clean will overwrite the CORE dataset.')
        filtered_core = core[~core['opinion_id'].isin(citing_missing_cited_ids)]
        filtered_core.to_parquet(_args.core)


def download_courtlistener_opinions_bulk():
    url = "https://www.courtlistener.com/api/bulk-data/opinions/all.tar"

    block_size = 1024  # 1 Kibibyte
    now = datetime.date.today()
    filename = os.path.join(_args.to, f'{now.strftime("%Y%m%d")}_opinion.tar')

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))

    logging.info(f'Download Court Listener utils.Opinion dataset, from URL {url}')
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

    @make_spin(Default, f'Unpacking archive {filename}')
    def _unpack():
        with tarfile.open(filename, 'r') as tar:
            for m in tqdm(tar.getmembers()):
                tar.extract(m, path=_args.to)
    _unpack()


_cited_dataset: Optional[OpinionDataset] = None


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
        new_parser.add_argument('--path', type=str, help='SOURCE dataset')
        new_parser.add_argument('--dest', type=str, help='DESTINATION dataset')
        if parallel:
            new_parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
            new_parser.add_argument('--batch-size', type=int, default=1024, help='Number of records sent at once to a '
                                                                                 'worker')
        new_parser.set_defaults(func=func)
        return new_parser

    # Download the utils.Opinion dataset from CourtListener
    # Unpack it if requested
    parser_download = subparsers.add_parser('download')
    parser_download.add_argument('--to', type=str, help='Download folder')
    parser_download.add_argument('--untar', default=False, action='store_true', help='Untar the archive')
    parser_download.set_defaults(func=download_courtlistener_opinions_bulk)

    # From CourtListener to Parquet dataset
    parser_to_parquet = default_parser(
        name='opinion',
        description='Convert the original dataset (a collection of tar.gz files in the '
                    'PATH folder), into a PARQUET dataset (multiple PARQUET files in the'
                    ' DEST folder',
        func=create_opinion_dataset
    )
    parser_to_parquet.add_argument('--tags', nargs='+', help='List of tags to get for each opinion JSON file '
                                                             'in each tar,gz file in folder')

    # Create a dataset of citing / cited opinion
    _ = default_parser(
        name='citation-map',
        description='Extract all citations from the PARQUET dataset of opinions located in PATH, and generate a '
                    'dataset of pairs citing_opinion / cited_opinion. The generated dataset is saved as a PARQUET'
                    'dataset in the folder DEST.',
        func=create_citation_map
    )

    # Random sample from the opinion dataset
    parser_sample = default_parser(
        name='sample',
        description='Pick a random sample of the opinions from the pool of all the opinions in the PARQUET dataset '
                    'in folder PATH. The sample is saved as a PARQUET dataset in folder DEST.',
        parallel=False,
        func=create_sample_dataset
    )
    parser_sample.add_argument('--citation-map', type=str, help='CSV File of the citation map')
    parser_sample.add_argument('--num-samples', type=int, default=10000, help='Number of samples')
    parser_sample.add_argument('--add-cited', default=False, action='store_true',
                               help='Add all the opinions that are cited by an opinion in the random sample. '
                                    'The size of the final sample will be larger than the argument --num-samples')
    parser_sample.add_argument('--add-citing', default=False, action='store_true',
                               help='Add all the opinions that cite an opinion in the random sample. The size of the '
                                    'final sample will be larger than the argument --num-samples')

    # Check a sample for completeness
    parser_verify = subparsers.add_parser(
        name='verify',
        description='Verify a random sample, to make sure all opinions cited from within the core sample are located '
                    'in the cited dataset.')
    parser_verify.add_argument('--core', type=str, help='Path to the PARQUET file with the CORE sample')
    parser_verify.add_argument('--cited', type=str, help='Path to the PARQUET file with the CITED sample')
    parser_verify.add_argument('--clean', default=False, action='store_true', help='Clean the CORE sample (will '
                                                                                   'overwrite its PARQUET file')
    parser_verify.set_defaults(func=verify_sample)

    # Produce data for annotation
    parser_doccano = default_parser(
        name='doccano',
        description='From an opinion PARQUET dataset located in folder PATH, extract for each citation a snippet of'
                    'text surrounding the citation for manual anchor annotation. The text snippets are saved as a '
                    'JSONL file in folder DEST',
        func=create_annotation_dataset
    )
    parser_doccano.add_argument('--max-words-extract', type=int, help='Limit the text around the citation '
                                                                      'itself to a number of characters')

    # Produce a summary of opinions
    parser_summary = default_parser(
        name='summary',
        description='From an opinion PARQUET dataset located in folder PATH, generate an extractive summary of each'
                    'opinion, using one of the available methods.',
        func=create_summary_dataset
    )
    parser_summary.add_argument('--method', type=str, choices=utils.summarization_methods.keys(), default='textrank',
                                help='Summarization method.')

    # Produce data for fairseq
    parser_prophet = default_parser(name='prophetnet',
                                    description='Generate file for ProphetNet summarization',
                                    func=create_prophetnet)
    parser_prophet.add_argument('--export', type=str, help='Copy the generated data into the ProphetNet folder for '
                                                           'original data')

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
