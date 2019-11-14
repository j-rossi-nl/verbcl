import glob
import os
import tarfile
import tqdm
import pandas as pd
import json
import re
import sys
import bs4
import csv
import multiprocessing
import en_core_web_sm

from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from argparse import ArgumentParser
from typing import List

from utils import multiprocess_courtlistener, create_destfilepath

# The arguments of the command are presented as a global module variable, so all functions require no arguments
args = None

# The SpaCy NLP engine
nlp = en_core_web_sm.load()

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
            for l in open(f):
                # Just take care each line terminates with a '\n'
                out.write(l)
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
                if not all([t in js for t in args.tags]):
                    continue
                data.append([js[t] for t in args.tags])
    except Exception as caught:
        print('Processing {}, Exception {}'.format(x, caught))
    csv_out = os.path.join(args.dest, '{}_{}.csv'.format(os.path.basename(x)[:-7], '_'.join(args.tags)))
    df = pd.DataFrame(data, columns=args.tags)
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
        df[args.tag] = df[args.tag].astype(str)
        txtfile_path = os.path.join(args.dest, '{}.txt'.format(os.path.basename(x)[:-4]))
        with open(txtfile_path, 'w') as txtfile:
            for _, d in df.iterrows():
                # Step 1 : Extract the full text from the HTML
                html = BeautifulSoup(d[args.tag], 'html.parser')
                txts = html.find_all(text=True)
                full_txt = ' '.join(txts)
                full_txt.replace('\n', '')

                # Step 2 : Split into sentences
                sentences = nlp(full_txt).sents
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
    print('Extracting Tags: {}\nFrom folder: {}'.format(' '.join(args.tags), args.targzpath))

    def extract_and_save_builder():
        return args.targzpath, 'tar.gz', args.dest, 'csv', extract_html
    results: List[str] = multiprocess_courtlistener(process_builder=extract_and_save_builder,
                                                    nbworkers=args.nbworkers)
    if args.gathercsv is not None:
        gathercsv(results, args.gathercsv, args.delete)


def clean_fk_id():
    """
    In COURT LISTENER database scheme, FK are often URI from which the ID has to be extracted (more compact)
    Does not work for all data (courts for example).
    :return:
    """
    print('Clean the FK ids from CSV files in folder: {}'.format(args.csvpath))
    match = re.compile(r'https://www\.courtlistener\.com(?::\d+)?/api/rest/v3/(?P<type>[a-z]+)/(?P<id>\d+)/')

    def extract_id(x):
        groups = match.search(x)
        if groups is not None:
            return int(groups.group('id'))
        return x

    for csv in tqdm.tqdm(glob.glob(os.path.join(args.csvpath, '*.csv'))):
        df = pd.read_csv(csv)
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
    print('Extracting all txt from csv in folder: {}'.format(args.csvpath))

    def prepare_bert_builder():
        return args.csvpath, 'csv', args.dest, 'txt', html_to_text
    _ = multiprocess_courtlistener(process_builder=prepare_bert_builder,
                                   nbworkers=args.nbworkers)


def process_citations(x):
    """
    Get a list of cited opinions. Takes in a CSV file with a column html_with_citations, extract from each opinion
    the list of citations, and generate a CSV file with 2 columns: citing_opinion_id, cited_opinion_id
    :param x: The name of the CSV file to process
    :return: The name of the generated CSV file
    """
    destcsv = os.path.join(args.dest, '{}_citations.csv'.format(os.path.basename(x)[:-4]))
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
        return args.csvpath, 'csv', args.dest, 'csv', process_citations
    results = multiprocess_courtlistener(process_builder=extract_citations_builder, nbworkers=args.nbworkers)
    if args.gathercsv is not None:
        gathercsv(results, args.gathercsv, remove=args.delete)


def produce_backref(x):
    """
    Produce a CSV file with 2 columns: opinion_id, csv_file to backtrack which opinion_id comes from which csv file

    :param x: a court CSV file
    :return: the path to a CSV file
    """
    destcsv = os.path.join(args.dest, '{}_backrefs.csv'.format(os.path.basename(x)[:-4]))
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
        return args.csvpath, 'csv', args.dest, 'csv', produce_backref
    results = multiprocess_courtlistener(process_builder=backreference_builder, nbworkers=args.nbworkers)
    if args.gathercsv is not None:
        gathercsv(results, args.gathercsv, remove=args.delete)


bs4_pagination_args = {'class': 'pagination'}
bs4_nolink_citation_args = {'class': 'citation no-link'}
bs4_citation_args = {'class': 'citation', 'data-id': True}
quote_in_sentence = re.compile(r'^.*"(?P<quote>[^"]+)".+')


def extract_gist(x: str):
    """
    Extract for each citation, the 'GIST' of it, which is the catchphrase that introduces the citED opinion
    in the citING opinion. Each cited opinion is introduced by a sentence that underlines what from the cited
    opinion is an argument for the citing opinion.

    :param x: the name of CSV file containing opinions
    :return: the path to a CSV file with columns citing_opinion_id, cited_opinion_id, extracted_gist
    """

    # Use args.destcsv as the folder where generated CSV files are saved, for file xxx.csv, the generated file
    # will be named xxx_gst.csv

    d = pd.read_csv(x).fillna('')

    opinion_gists = []
    for _, opinion in tqdm.tqdm(d.iterrows(), total=len(d), desc=os.path.basename(x), leave=False,
                                position=int(multiprocessing.current_process().name.split('-')[1]),
                                bar_format='{desc:<20}{percentage:3.0f}%|{bar}| {n_fmt:>6}/{total_fmt:>6} '
                                           '[{remaining:>10}]'):
        citing_opinion_id = int(opinion['opinion_id'])
        html = BeautifulSoup(opinion['html_with_citations'], 'html.parser')

        # A bit of cleaning on the original HTML
        # It includes tags for original pagination that insert numbers here and there in the text
        for page in html.find_all('span', **bs4_pagination_args):
            page: bs4.element.Tag
            page.decompose()

        processed_ids = []
        for cited in html.find_all('span', **bs4_citation_args):
            cited: bs4.element.Tag

            # One citation can generate multiple spans in the document. Guess it is an artefact of the html generation.
            # We keep track of which ones have been processed.
            # TODO : it means that we restrict to the first time an opinion is cited, by interpreting multiple
            # TODO : citations as an incorrect HTML generation
            cited_opinion_id = cited['data-id']
            if cited_opinion_id in processed_ids:
                continue
            processed_ids.append(cited_opinion_id)

            # The catchphrase will be before the citation in the text, close by, and will be a full sentence
            # As a heuristic, we go through the previous siblings in the HTML parsing and retain the first one
            # with more than 10 words
            # TODO : smarter detection
            candidate_text = ''
            for candidate_sibling in cited.previous_siblings:
                if isinstance(candidate_sibling, bs4.element.Tag):
                    candidate_text = candidate_sibling.text
                else:
                    candidate_text = candidate_sibling
                if len(candidate_text.split()) > 10:
                    break

            # We focus on the last full sentence of the candidate text. Again we select, from the last sentence,
            # the sentence with more than 10 words
            candidate_sentence = ''
            for s in reversed(sent_tokenize(candidate_text)):
                if len(s.split()) > 10:
                    candidate_sentence = s
                    break

            # Now in the selected text before the citation, we retain either the last full sentence, or a text
            # text enclosed in quotemarks in the last full sentence
            # For example: J could ask for ... against ... as in G v R <citation> --> full sentence
            # Or: J could ask ... as "... ... ..." in G vR <citation>             --> only what's in quotes
            m = quote_in_sentence.match(candidate_sentence)
            op_gist = ''
            if m is None:
                op_gist = candidate_sentence.strip()
            else:
                op_gist = m.group('quote')

            opinion_gists.append([citing_opinion_id, cited_opinion_id, op_gist])

    # Wrap-up, save to CSV file and return the name of that CSV file to caller
    df = pd.DataFrame(opinion_gists, columns=['citing_opinion_id', 'cited_opinion_id', 'gist'])
    if args.json:
        dest = create_destfilepath(x, args.dest, 'gist', new_extension='json')
        df.to_json(dest, orient='records', lines=True)
    else:
        dest = create_destfilepath(x, args.dest, 'gist')
        df.to_csv(dest, index=False, quoting=csv.QUOTE_ALL)

    return dest


def gist():
    def gist_builder():
        return args.csvpath, 'csv', args.dest, 'json' if args.json else 'csv', extract_gist
    results = multiprocess_courtlistener(process_builder=gist_builder, nbworkers=args.nbworkers)
    if args.gather is not None:
        if args.json:
            gatherjson(results, args.gather, remove=args.delete)
        else:
            gathercsv(results, args.gather, remove=args.delete)


def parse_args(argstxt=sys.argv[1:]):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

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

    parser_cleanid = subparsers.add_parser('csvcleanid')
    parser_cleanid.add_argument('--csvpath', type=str, help='Folder with all CSV files to post process')
    parser_cleanid.set_defaults(func=clean_fk_id)

    parser_prepare_bert = subparsers.add_parser('preparebert')
    parser_prepare_bert.add_argument('--csvpath', type=str, help='Folder with opinions CSV files, as extracted with'
                                                                 'command targztocsv, the tag given in --tag'
                                                                 'has to be extracted')
    parser_prepare_bert.add_argument('--tag', type=str, help='Tag that contains the text')
    parser_prepare_bert.add_argument('--dest', type=str, help='Destination folder for the TXT files')
    parser_prepare_bert.add_argument("--nbworkers", type=int, default=4, help="Number of parallel workers")
    parser_prepare_bert.set_defaults(func=prepare_bert)

    parser_gist = subparsers.add_parser('gist')
    parser_gist.add_argument('--csvpath', type=str, help='Path to the folder with all csv files coming from'
                                                         'running targztocsv with tags containing '
                                                         'html_with_citations')
    parser_gist.add_argument('--gather', type=str, help='Destination single CSV where all data from generated'
                                                           'CSV is merged')
    parser_gist.add_argument('--dest', type=str, help='Destination folder for CSV files')
    parser_gist.add_argument('--json', action='store_true', help='Store as JSON instead of CSV')
    parser_gist.add_argument('--delete', action='store_true', help='Delete generated CSV, except the gathercsv')
    parser_gist.add_argument('--nbworkers', type=int, default=4, help='Number of parallel workers')
    parser_gist.set_defaults(func=gist)

    global args
    args = parser.parse_args(argstxt)


def main():
    parse_args()
    args.func()


if __name__ == '__main__':
    main()
#    try:
#    except Exception as excp:
#        import pdb
#        pdb.post_mortem()
