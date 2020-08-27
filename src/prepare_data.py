import glob
import os
import tarfile
import tqdm
import pandas as pd
import json
import re
import sys
import csv
import multiprocessing
import en_core_web_sm
import pickle
import gensim

from argparse import ArgumentParser
from typing import List, Dict, Callable
from multiprocessing import Queue
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from utils import multiprocess_courtlistener, create_destfilepath, multiprocess_with_queue, df_from_file
from gist_extraction import extract_catchphrase

# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args = None

# The SpaCy NLP engine
_nlp = en_core_web_sm.load()


def gathercsv(csvfiles, csvdest, remove=True):
    df = pd.concat([pd.read_csv(x) for x in csvfiles], ignore_index=True)
    df.to_csv(csvdest, index=False)
    if remove:
        for tmpf in csvfiles:
            os.remove(tmpf)


def gatherjson(jsonfiles, jsondest, remove=True):
    """
    Get all generated JSON files and aggregate in one single JSON file
    In our case, we use orient='records' and list=True, so it is just a matter of concatenating files

    :param jsonfiles:
    :param jsondest:
    :param remove:
    :return:
    """
    with open(jsondest, 'w') as out:
        for f in jsonfiles:
            with open(f) as js:
                for line in js:
                    # Just take care each line terminates with a '\n'
                    out.write(line)
            out.write('\n')
    if remove:
        for tmpf in jsonfiles:
            os.remove(tmpf)


def extract_html(x):
    """
    Open a targz file, go through all contained JSON files, without extracting to disk, and in each file, extract
    a set of values, based on the list of tags given in args.tags. Builds a CSV file with all extracted data, one line
    per original JSON file.
    The CSV file is in the args.dest folder.
    Assumes the targz is made only of json files (it is the case in COURT LISTENER dataset)

    :param x: path to targz file
    :return: the path to the CSV file
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
    csv_out = os.path.join(_args.dest, '{}_{}.csv'.format(os.path.basename(x)[:-7], '_'.join(_args.tags)))
    df = pd.DataFrame(data, columns=_args.tags)
    df.to_csv(csv_out, index=False)
    return csv_out


def html_to_text(x):
    """
    Extract all texts from a CSV file, using the column args.tag, and outputs the raw text, split in sentences, to a
    TXT file, with identical name in the same folder.

    :param x: the path of the CSV file
    :return: the path of the TXT file
    """
    txtfile_path = ''
    try:
        df = pd.read_csv(x)
        df[_args.tag] = df[_args.tag].astype(str)
        txtfile_path = os.path.join(_args.dest, '{}.txt'.format(os.path.basename(x)[:-4]))
        with open(txtfile_path, 'w') as txtfile:
            for _, d in df.iterrows():
                # Step 1 : Extract the full text from the HTML
                html = BeautifulSoup(d[_args.tag], 'html.parser')
                txts = html.find_all(text=True)
                full_txt = ' '.join(txts)
                full_txt.replace('\n', '')

                # Step 2 : Split into sentences
                sentences = _nlp(full_txt).sents
                for s in sentences:
                    txtfile.write('{}\n'.format(s))
                txtfile.write('\n')
    except Exception as caught:
        print('Processing {}, Exception {}'.format(x, caught))
    return txtfile_path


def extract_and_save():
    """
    Extract values from JSON files in targz files. Each targz file will generate a CSV file.
    If needed, gather all CSV files into one large CSV file, based on args.gathercsv, and delete all
    intermediary CSV files if args.delete.

    :return:
    """
    print('Extracting Tags: {}\nFrom folder: {}'.format(' '.join(_args.tags), _args.targzpath))

    def extract_and_save_builder():
        return _args.targzpath, 'tar.gz', _args.dest, 'csv', extract_html

    results: List[str] = multiprocess_courtlistener(process_builder=extract_and_save_builder,
                                                    nbworkers=_args.nbworkers)
    if _args.gathercsv is not None:
        gathercsv(results, _args.gathercsv, _args.delete)


def clean_fk_id():
    """
    In COURT LISTENER database scheme, FK are often URI from which the ID has to be extracted (more compact)
    Does not work for all data (courts for example).
    :return:
    """
    print('Clean the FK ids from CSV files in folder: {}'.format(_args.csvpath))
    match = re.compile(r'https://www\.courtlistener\.com(?::\d+)?/api/rest/v3/(?P<type>[a-z]+)/(?P<id>\d+)/')

    def extract_id(x):
        groups = match.search(x)
        if groups is not None:
            return int(groups.group('id'))
        return x

    for csv_file in tqdm.tqdm(glob.glob(os.path.join(_args.csvpath, '*.csv'))):
        df = pd.read_csv(csv_file)
        uri_column = [c for c in df.columns if c not in ['id']][0]
        cleanid_column = '{}_id'.format(uri_column)
        df[cleanid_column] = df[uri_column].apply(extract_id)
        df[['id', cleanid_column]].to_csv('{}_clean.csv'.format(csv[:-4]), index=False)


def prepare_bert():
    """
    Go through the texts of all opinions and generate data in a format suitable for input to BERT pre-training
    generation.
    Goes through all CSV files in a folder, extract and normalize text from args.tag, and outputs a TXT
    file with normalized sentences.

    :return:
    """
    print('Extracting all txt from csv in folder: {}'.format(_args.csvpath))

    def prepare_bert_builder():
        return _args.csvpath, 'csv', _args.dest, 'txt', html_to_text

    _ = multiprocess_courtlistener(process_builder=prepare_bert_builder,
                                   nbworkers=_args.nbworkers)


def process_citations(x):
    """
    Get a list of cited opinions. Takes in a CSV file with a column html_with_citations, extract from each opinion
    the list of citations, and generate a CSV file with 2 columns: citing_opinion_id, cited_opinion_id
    :param x: The name of the CSV file to process
    :return: The name of the generated CSV file
    """
    destcsv = os.path.join(_args.dest, '{}_citations.csv'.format(os.path.basename(x)[:-4]))
    d = pd.read_csv(x)
    citations = []
    for _, opinion in d.iterrows():
        citing_opinion_id = int(opinion['opinion_id'])
        html = opinion['html_with_citations'] if type(opinion['html_with_citations']) == str else ''
        bs = BeautifulSoup(html, 'html.parser')
        search = {'class': 'citation', 'data-id': True}
        cited_opinions = set([int(x['data-id']) for x in bs.find_all('span', **search)])
        citations.extend([[citing_opinion_id, x] for x in cited_opinions])

    pd.DataFrame(citations, columns=['citing_opinion_id', 'cited_opinion_id']).to_csv(destcsv, index=False)
    return destcsv


def extract_citations():
    def extract_citations_builder():
        return _args.csvpath, 'csv', _args.dest, 'csv', process_citations

    results = multiprocess_courtlistener(process_builder=extract_citations_builder, nbworkers=_args.nbworkers)
    if _args.gathercsv is not None:
        gathercsv(results, _args.gathercsv, remove=_args.delete)


def produce_backref(x):
    """
    Produce a CSV file with 2 columns: opinion_id, csv_file to backtrack which opinion_id comes from which csv file

    :param x: a court CSV file
    :return: the path to a CSV file
    """
    destcsv = os.path.join(_args.dest, '{}_backrefs.csv'.format(os.path.basename(x)[:-4]))
    d = pd.read_csv(x)
    csv_file = os.path.basename(x)
    backrefs = []
    for _, opinion in d.iterrows():
        opinion_id = int(opinion['opinion_id'])
        backrefs.append([opinion_id, csv_file])

    pd.DataFrame(backrefs, columns=['opinion_id', 'csv_file']).to_csv(destcsv, index=False)
    return destcsv


def backreference():
    def backreference_builder():
        return _args.csvpath, 'csv', _args.dest, 'csv', produce_backref

    results = multiprocess_courtlistener(process_builder=backreference_builder, nbworkers=_args.nbworkers)
    if _args.gathercsv is not None:
        gathercsv(results, _args.gathercsv, remove=_args.delete)


def extract_gist(in_queue: Queue, out_queue: Queue) -> None:
    """
    Extract for each citation, the 'GIST' of it, which is the catchphrase that introduces the citED opinion
    in the citING opinion. Each cited opinion is introduced by a sentence that underlines what from the cited
    opinion is an argument for the citing opinion.

    :param in_queue: tasks to do
    :param out_queue: done tasks
    :return: the path to a CSV file with columns citing_opinion_id, cited_opinion_id, extracted_gist
    """
    # Use args.destcsv as the folder where generated CSV files are saved, for file xxx.csv, the generated file
    # will be named xxx_gst.csv
    while True:
        x: str = in_queue.get(True)
        d = pd.read_csv(x).fillna('').set_index('opinion_id')

        opinion_gists = []
        if _args.monitor:
            iterator = tqdm.tqdm(d.iterrows(), total=len(d), desc=os.path.basename(x),
                                 leave=False, position=int(multiprocessing.current_process().name.split('-')[1]),
                                 bar_format='{desc:<20}{percentage:3.0f}%|{bar}| {n_fmt:>6}/{total_fmt:>6} '
                                            '[{rate_fmt:>4} {remaining:>6}]')
        else:
            iterator = d.iterrows()
        for citing_opinion_id, opinion in iterator:
            op_gists = extract_catchphrase(opinion['html_with_citations'], method=_args.method)
            opinion_gists.extend([[citing_opinion_id, g['cited_opinion_id'], g['gist']] for g in op_gists])

        # Wrap-up, save to CSV file and return the name of that CSV file to caller
        df = pd.DataFrame(opinion_gists, columns=['citing_opinion_id', 'cited_opinion_id', 'gist'])
        if _args.json:
            dest = create_destfilepath(x, _args.dest, 'gist', new_extension='json')
            df.to_json(dest, orient='records', lines=True)
        else:
            dest = create_destfilepath(x, _args.dest, 'gist')
            df.to_csv(dest, index=False, quoting=csv.QUOTE_ALL)

        out_queue.put(dest)


def gist():
    def _opinions_csv_iterator():
        for p in glob.glob(os.path.join(_args.csvpath, '*.csv')):
            yield p

    results = multiprocess_with_queue(worker_fn=extract_gist,
                                      input_iterator_fn=_opinions_csv_iterator,
                                      nb_workers=_args.nbworkers,
                                      description='Gist')

    if _args.gather is not None:
        if _args.json:
            gatherjson(results, _args.gather, remove=_args.delete)
        else:
            gathercsv(results, _args.gather, remove=_args.delete)


def sort_gist_json():
    d: pd.DataFrame = pd.read_json(_args.json, orient='records', lines=True)
    d.sort_values(by='cited_opinion_id', ascending=True, inplace=True)
    d.to_json(_args.json, orient='records', lines=True)


def tfidf():
    """
    Randomly draws a corpus of opinions. Trains a TFIDF Vectorizer on it.

    :return:
    """
    backrefs = pd.read_csv(_args.backrefs)
    draw: pd.DataFrame = backrefs.sample(n=_args.draw)

    def _iter_corpus():
        for _, r in tqdm.tqdm(draw.iterrows(), total=len(draw)):
            d = pd.read_csv(os.path.join(_args.csvpath, r['csv_file'])).set_index('opinion_id').fillna('')
            soup = BeautifulSoup(d.loc[r['opinion_id']]['html_with_citations'], 'html.parser')
            txt = ' '.join(soup.find_all(text=True))
            yield txt

    vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, min_df=5)
    vectorizer.fit(_iter_corpus())

    pickle.dump(vectorizer, open(_args.save, 'wb'))


def split_worker(in_queue: Queue, out_queue: Queue) -> None:
    """
    Get each CSV file and split it into smaller files, each with max args.size number of records

    :param in_queue: tasks to do
    :param out_queue: done tasks
    :return: the list of smaller files generated
    """
    # Use args.destcsv as the folder where generated CSV files are saved, for file xxx.csv, the generated files
    # will be named xxx_0001.csv
    while True:
        x: str = in_queue.get(True)
        d = pd.read_csv(x)

        nb_records = len(d)
        nb_chunks = nb_records // _args.size
        nb_chunks += 1 if nb_records % _args.size > 0 else 0
        chunks = [{
            'file': create_destfilepath(x, _args.dest, addsuffix='{:04d}'.format(i)),
            'data': d[i * _args.size:(i + 1) * _args.size]
        } for i in range(nb_chunks)]

        for chunk in chunks:
            data: pd.DataFrame = chunk['data']
            data.to_csv(chunk['file'], index=False)

        out_queue.put([c['file'] for c in chunks])


def split():
    def _opinions_csv_iterator():
        for p in glob.glob(os.path.join(_args.csvpath, '*.csv')):
            yield p

    _ = multiprocess_with_queue(worker_fn=split_worker,
                                input_iterator_fn=_opinions_csv_iterator,
                                nb_workers=_args.nbworkers,
                                description='Split')


def summarize_textrank_worker(in_queue: Queue, out_queue: Queue) -> None:
    """
    Applies textrank to summarize a collection of texts
    """
    while True:
        x: Dict = in_queue.get()
        uuid: int = x['uuid']
        text: str = x['text']
        summary: str = gensim.summarization.summarizer.summarize(text, word_count=_args.nbwords)

        out_queue.put({'opinion_id': uuid, 'summary': summary})


def opinion_summarize():
    """
    Uses all texts from the CSV file args.texts (column args.tag), summarize acording to args.method and outputs a
    CSV file with fields 'opinion_id' and 'summary'
    """
    method_name_2_func: Dict[str, Callable[[Queue, Queue], None]] = {
        'textrank': summarize_textrank_worker,
    }

    def _opinions_text_iterator():
        d = df_from_file(_args.texts).set_index('opinion_id')
        for uuid, data in d.iterrows():
            yield {'uuid': uuid, 'text': data[_args.tag]}

    results = multiprocess_with_queue(worker_fn=method_name_2_func[_args.method],
                                      input_iterator_fn=_opinions_text_iterator,
                                      nb_workers=_args.nbworkers,
                                      description=_args.method)

    df = pd.DataFrame(results, columns=['opinion_id', 'summary'])
    df.to_csv(_args.dest, index=False)


def gists_summarize():
    """
    Uses all texts from the CSV file args.texts (column args.tag), summarize acording to args.method and outputs a
    CSV file with fields 'opinion_id' and 'summary'
    """
    method_name_2_func: Dict[str, Callable[[Queue, Queue], None]] = {
        'textrank': summarize_textrank_worker,
    }

    def _opinions_text_iterator():
        # Each cited opinion has many gists, we want to merge them as a text to summarize
        d = df_from_file(_args.texts)
        groups = d.groupby(_args.uuid)

        # Join together all extracted gists
        for uuid, gists in groups:
            yield {'uuid': uuid, 'text': ' . '.join(gists[_args.tag])}

    results = multiprocess_with_queue(worker_fn=method_name_2_func[_args.method],
                                      input_iterator_fn=_opinions_text_iterator,
                                      nb_workers=_args.nbworkers,
                                      description=_args.method)

    df = pd.DataFrame(results, columns=['opinion_id', 'summary'])
    df.to_csv(_args.dest, index=False)


def parse_args(argstxt=None):
    if argstxt is None:
        argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    # Extract the opinions from the tar.gz court files, and create a record per opinion containing a list of tags
    parser_tocsv = subparsers.add_parser('targztocsv')
    parser_tocsv.add_argument("--targzpath", type=str, help="The path to the folder with all *.tar.gz "
                                                            "files to be decompressed")
    parser_tocsv.add_argument('--dest', type=str, help='Destination folder for the CSV files')
    parser_tocsv.add_argument('--gathercsv', type=str, help='Destination file for  the extracted data')
    parser_tocsv.add_argument('--tags', nargs='+', help='List of tags to get for each JSON file '
                                                        'in each tar,gz file in folder')
    parser_tocsv.add_argument('--delete', action='store_true', help='Delete the intermediary CSV files')
    parser_tocsv.add_argument("--nbworkers", type=int, default=4, help="Number of parallel workers")
    parser_tocsv.set_defaults(func=extract_and_save)

    # Create a CSV file to track all couples of (citing, cited) opinions
    parser_citations = subparsers.add_parser('citations')
    parser_citations.add_argument('--csvpath', type=str, help='Path to the folder with all csv files coming from'
                                                              'running targztocsv with tags containing '
                                                              'html_with_citations')
    parser_citations.add_argument('--dest', type=str, help='Destination folder for CSV files')
    parser_citations.add_argument('--gathercsv', type=str, help='Destination single CSV where all data from generated'
                                                                'CSV is merged')
    parser_citations.add_argument('--delete', action='store_true', help='Delete generated CSV, except the gathercsv')
    parser_citations.add_argument('--nbworkers', type=int, default=4, help='Number of parallel workers')
    parser_citations.set_defaults(func=extract_citations)

    # Create a CSV file to track which opinion is in which court CSV file
    parser_backref = subparsers.add_parser('backref')
    parser_backref.add_argument('--csvpath', type=str, help='Path to the folder with all csv files coming from'
                                                            'running targztocsv with tags containing '
                                                            'html_with_citations')
    parser_backref.add_argument('--gathercsv', type=str, help='Destination single CSV where all data from generated'
                                                              'CSV is merged')
    parser_backref.add_argument('--dest', type=str, help='Destination folder for CSV files')
    parser_backref.add_argument('--delete', action='store_true', help='Delete generated CSV, except the gathercsv')
    parser_backref.add_argument('--nbworkers', type=int, default=4, help='Number of parallel workers')
    parser_backref.set_defaults(func=backreference)

    # I don't know what it does
    parser_cleanid = subparsers.add_parser('csvcleanid')
    parser_cleanid.add_argument('--csvpath', type=str, help='Folder with all CSV files to post process')
    parser_cleanid.set_defaults(func=clean_fk_id)

    # Extract raw text from the opinions
    parser_prepare_bert = subparsers.add_parser('preparebert')
    parser_prepare_bert.add_argument('--csvpath', type=str, help='Folder with opinions CSV files, as extracted with'
                                                                 'command targztocsv, the tag given in --tag'
                                                                 'has to be extracted')
    parser_prepare_bert.add_argument('--tag', type=str, help='Tag that contains the text')
    parser_prepare_bert.add_argument('--dest', type=str, help='Destination folder for the TXT files')
    parser_prepare_bert.add_argument("--nbworkers", type=int, default=4, help="Number of parallel workers")
    parser_prepare_bert.set_defaults(func=prepare_bert)

    # Split the court CSV files into something smaller
    parser_split_csv = subparsers.add_parser('splitcsv')
    parser_split_csv.add_argument('--csvpath', type=str, help='Folder with original CSV files')
    parser_split_csv.add_argument('--dest', type=str, help='Folder that will receive the split files')
    parser_split_csv.add_argument('--size', type=int, default=500, help='Number of records per CSV file')
    parser_split_csv.add_argument('--nbworkers', type=int)
    parser_split_csv.set_defaults(func=split)

    # Extract all GIST from all citations
    parser_gist = subparsers.add_parser('gist')
    parser_gist.add_argument('--method', type=str, choices=['nlp', 'last'], default='last',
                             help='Select the method to extract gist.')
    parser_gist.add_argument('--csvpath', type=str, help='Path to the folder with all csv files coming from'
                                                         'running targztocsv with tags containing '
                                                         'html_with_citations')
    parser_gist.add_argument('--gather', type=str, help='Destination single CSV where all data from generated'
                                                        'CSV is merged')
    parser_gist.add_argument('--dest', type=str, help='Destination folder for JSON files')
    parser_gist.add_argument('--json', action='store_true', help='Store as JSON instead of CSV')
    parser_gist.add_argument('--delete', action='store_true', help='Delete generated CSV, except the gathercsv')
    parser_gist.add_argument('--monitor', action='store_true', help='Have a progress bar per worker')
    parser_gist.add_argument('--nbworkers', type=int, default=4, help='Number of parallel workers')
    parser_gist.set_defaults(func=gist)

    # Prepare a TFIDF Vectorizer based on a random sample from the corpus
    parser_tfidf = subparsers.add_parser('tfidf')
    parser_tfidf.add_argument('--csvpath', type=str, help='Path to CSV files')
    parser_tfidf.add_argument('--save', type=str, help='File to pickle the tfidf vectorizer')
    parser_tfidf.add_argument('--draw', type=int, help='Number of opinions to draw randomly')
    parser_tfidf.add_argument('--backrefs', type=str, help='Path to backrefs CSV file')
    parser_tfidf.set_defaults(func=tfidf)

    # Sort the big JSON file with all gists by cited_opinion_id
    parser_sort = subparsers.add_parser('sort')
    parser_sort.add_argument('--json')
    parser_sort.set_defaults(func=sort_gist_json)

    # Summarize each opinion
    parser_opsum = subparsers.add_parser('opinion_summary')
    parser_opsum.add_argument('--texts', type=str, help='Path to the CSV file with all texts')
    parser_opsum.add_argument('--tag', type=str, default='text', help='Column name for the text')
    parser_opsum.add_argument('--dest', type=str, help='Path to the result CSV file')
    parser_opsum.add_argument('--method', type=str, choices=['textrank'], help='Summarization technic')
    parser_opsum.add_argument('--nbwords', type=int, default=200)
    parser_opsum.add_argument('--nbworkers', type=int, help='Number of parallel workers')
    parser_opsum.set_defaults(func=opinion_summarize)

    # Summarize each opinion from the gists
    parser_gisum = subparsers.add_parser('gists_summary')
    parser_gisum.add_argument('--texts', type=str, help='Path to the CSV file with all texts')
    parser_gisum.add_argument('--tag', type=str, default='text', help='Column name for the text')
    parser_gisum.add_argument('--uuid', type=str, default='text', help='Column name for the UUID')
    parser_gisum.add_argument('--dest', type=str, help='Path to the result CSV file')
    parser_gisum.add_argument('--method', type=str, choices=['textrank'], help='Summarization technic')
    parser_gisum.add_argument('--nbwords', type=int, default=200)
    parser_gisum.add_argument('--nbworkers', type=int, help='Number of parallel workers')
    parser_gisum.set_defaults(func=gists_summarize)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()


if __name__ == '__main__':
    main()
#    try:
#    except Exception as excp:
#        import pdb
#        pdb.post_mortem()
