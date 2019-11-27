import os
import pandas as pd
import sys
import torch
import pickle
import tqdm
import numpy as np
import json
import logging

from multiprocessing import Queue
from transformers import BertTokenizer, BertModel
from argparse import ArgumentParser
from typing import Dict
from utils import multiprocess_with_queue

# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BERT_MODEL = 'bert-base-uncased'


def extract_opinion():
    """
    Extract one single opinion, identified by its opinion_id args.opinion_id. It uses the backrefs file, generated from
    prepare_data backrefs. It looks for the court CSV file in the folder args.csvpath.
    The HTML of this opinion is then saved in the folder args.htmlpath
    :return:
    """
    opinion_id = _args.opinion_id[0]
    print('Reading index...')
    # Backrefs is a CSV file with pairs opinion_id -> court CSV filename (1051, 'wis.csv')
    back = pd.read_csv(_args.backrefs).set_index('opinion_id')
    print('Find in index...')
    csvfile = back.loc[opinion_id]['csv_file']

    # In the court CSV file we can get the HTML of this opinion
    csvpath = os.path.join(_args.csvpath, csvfile)
    print('Load CSV... (size = {:,})'.format(os.path.getsize(csvpath)))
    d = pd.read_csv(csvpath).set_index('opinion_id')   # inefficient... TODO lazy reading
    print('Read HTML...')
    html = d.loc[opinion_id]['html_with_citations']

    # Save the HTML to a file
    htmlpath = os.path.join(_args.htmlpath, '{}.html'.format(opinion_id))
    print('Save HTML to {}'.format(htmlpath))
    with open(htmlpath, 'w') as htmlfile:
        htmlfile.write(html)
    print('Done.')


def vecs():
    """
    Generate BERT embeddings for a collection of texts. The texts are already tokenized and are cached in file
    args.cache_tokens. Use the command tokens to generate this file.
    It saves a file <opinion_id>.vecs for each opinion_id. This file contains a list of 1-D np array vectors for all
    texts attached to the opinion_id.
    1 item of that list = vector for 1 extracted gist of the cited opinion. Order is preserved.
    :return:
    """
    # Use CUDA is possible - BERT on CPU is no-no !!
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    logger.info("Using device {}".format(device_name.upper()))

    # Load the tokens
    data = pickle.load(file=open(_args.cache_tokens, 'rb'))
    tokens_tensor, segments_tensor, all_uuids = data['tokens'], data['segments'], data['uuid']
    tokens_tensor: torch.Tensor
    segments_tensor: torch.Tensor
    all_uuids: np.ndarray

    logger.info('Load tokenized texts from file {}'.format(_args.cache_tokens))

    # Before running through the model, we split the data in batches
    tokens_batches = tokens_tensor.split(_args.batch_size)
    segments_batches = segments_tensor.split(_args.batch_size)

    model = BertModel.from_pretrained(BERT_MODEL)

    # Whenever, use all available GPUs
    # In multi GPU, batch_size can be very high (4096 for example). DataParallel takes care of running smoothly
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info('Using {} Parallel GPUs'.format(torch.cuda.device_count()))
    model.to(device)
    model.eval()

    # Now we run each batch, using BERT as a feature generator
    with torch.no_grad():
        np_outs = []
        for batch_tokens, batch_segments in tqdm.tqdm(zip(tokens_batches, segments_batches),
                                                      total=len(tokens_batches),
                                                      desc='BERT'):
            batch_tokens: torch.Tensor
            batch_segments: torch.Tensor

            # See the models docstrings for the detail of the inputs
            outputs = model(batch_tokens.to(device), token_type_ids=batch_segments.to(device))

            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded = outputs[1]   # POOLER_OUTPUT : shape (batch_size, hidden_size)
            np_outs.append(encoded.cpu().numpy())

        vectors = np.vstack(np_outs)

    # They have to be aligned at the beginning. Order is preserved.
    assert all_uuids.shape[0] == vectors.shape[0]

    # We rebuild the output based on the uuid vector
    # The inference has preserved the input ordering
    # results will be a dictionary with opinion_id as key. The values will be list of vectors
    results = {}
    for opinion_id, gist_vec in zip(all_uuids, vectors):
        if opinion_id in results:
            results[opinion_id].append(gist_vec)
        else:
            results[opinion_id] = [gist_vec]

    for opinion_id, gist_vecs in results.items():
        with open(os.path.join(_args.features, '{}.vecs'.format(opinion_id)), 'wb') as out:
            pickle.dump(np.vstack(gist_vecs), out)


def search_opinions():
    """
    Based on the arguments, search the cited opinions to extract.
    See the argparse documentation for the command 'tokens'
    :return:
    """
    # We can't load big files into memory. For those ones we'll do lazy reading
    # WARNING: in case of lazy reading, the file has to be sorted by cited_opinion_id !!
    lazy_reading = False
    if os.path.getsize(_args.gist) > 200e6:
        lazy_reading = True
        logger.info('Using Lazy Reading')

    # A dictionary with each extracted cited opinion along with a list [] of gists
    opinions = {}

    # Only one opinion was requested
    if _args.cited_opid is not None:
        if lazy_reading:
            # Line by line in the file
            opinions[_args.cited_opid] = []
            with open(_args.gist, 'r') as f:
                for l in f:
                    js = json.loads(l)
                    if int(js['cited_opinion_id']) != _args.cited_opid:
                        continue
                    opinions[_args.args.cited_opid].append(js['gist'])
        else:
            # load at once
            d = pd.read_json(_args.gist, orient='records', lines=True).set_index('cited_opinion_id')
            d: pd.DataFrame
            opinions[_args.cited_opid] = d.loc[_args.cited_opid]['gist'].values
    # In the case we want all opinions with a citation humber higher than the cut
    elif _args.min is not None:
        if lazy_reading:
            # Line by line in the file
            with open(_args.gist, 'r') as f:
                current_opinion_id = 0
                collected_lines = []
                for l in f:
                    js = json.loads(l)
                    if int(js['cited_opinion_id']) == current_opinion_id:
                        # Still reading about the same cited opinion
                        collected_lines.append(js)
                    else:
                        # Close the current cited opinion
                        if _args.min <= len(collected_lines) <= _args.max:
                            # Enough citations to make the cut
                            # we store the data and continue
                            opinions[current_opinion_id] = [x['gist'] for x in collected_lines]

                        collected_lines = []
                        current_opinion_id = js['cited_opinion_id']
        else:
            d = pd.read_json(_args.gist, orient='records', lines=True).set_index('cited_opinion_id')
            d: pd.DataFrame
            counts = d.index.value_counts()
            opinion_ids = counts[(counts >= _args.min) & (counts <= _args.max)].index
            for opinion_id in opinion_ids:
                opinions[opinion_id] = d.loc[opinion_id]['gist'].values

    logger.info('Identified {} cited opinions'.format(len(opinions)))
    return opinions


def _process_tokenize(in_queue: Queue, out_queue: Queue) -> None:
    """
    Worker process. Tokenize a collection of text. Reads from a 'to do' queue and sends results in a 'done' queue
    The queue item is a dict with keys 'uuid': the opinion_id, 'texts': a list of texts attached to this uuid
    It issues a dict with 3 keys: 'uuid', 'tokens': a list of list of tokens, padded to args.max_seq_len. and
    'segments': a list of list of segment_ids (in this case it's all 0), padded to args.max_seq_len.
    See BERT for the description of segment_id.

    :param in_queue: The queue with the tasks to do
    :param out_queue: the result from tokenization
    :return:
    """
    # Avoid the logging of 'loading message' for the Tokenizer
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    # Step 1 - prepare all inputs
    max_seq_len = _args.max_seq_len

    while True:
        x: Dict = in_queue.get(True)
        uuid = x['uuid']
        texts = x['texts']

        # We use BERT as feature generator for the sentences of gists
        # To make the texts edible by BERT, we have to tokenize, provide a segment id (in our case, it is always 0,
        # as there is only one sequence that we want to encode
        # all sequences must be padded to be the same length
        raw_tokens = [
            tokenizer.build_inputs_with_special_tokens(tokenizer.encode(t, max_length=max_seq_len - 2))
            for t in texts
        ]
        pad_tokens = np.vstack([np.array(t + [0] * (max_seq_len - len(t)), dtype=int)
                                for t in raw_tokens])
        segments = np.zeros((pad_tokens.shape[0], max_seq_len))
        out_queue.put({'uuid': uuid, 'tokens': pad_tokens, 'segments': segments})


def tokens():
    """
    Parent process.
    From a given corpus of gists, we get all those corresponding to an opinion
    Then we produce a feature vector for each gist with BERT that will be used later.
    This collection is pickled as a numpy array.
    :return:
    """
    opinions = search_opinions()

    def _opinions_iterator():
        for k, v in opinions.items():
            yield {'uuid': k, 'texts': v}

    collected_arrays = multiprocess_with_queue(worker_fn=_process_tokenize,
                                               input_iterator_fn=_opinions_iterator,
                                               nb_workers=_args.nb_workers,
                                               description='Tokenization')

    # Assemble all data and pickle
    # All np arrays should have the same first dimension, and be aligned
    all_tokens = np.vstack([x['tokens'] for x in collected_arrays])      # (nb_texts, max_seq_len)
    all_segments = np.vstack([x['segments'] for x in collected_arrays])  # (nb_texts, max_seq_len)
    all_uuids = np.hstack([[x['uuid']] * len(x['tokens']) for x in collected_arrays])         # (nb_texts,)

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    segments_tensor = torch.tensor(all_segments, dtype=torch.long)

    pickle.dump(obj={'tokens': tokens_tensor, 'segments': segments_tensor, 'uuid': all_uuids},
                file=open(_args.cache_tokens, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Saved tokenized texts to file {}'.format(_args.cache_tokens))


def parse_args(argstxt=sys.argv[1:]):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    parser_opinion = subparsers.add_parser('opinion')
    parser_opinion.add_argument('opinion_id', type=int, nargs=1)
    parser_opinion.add_argument('--backrefs', type=str, default='/ivi/ilps/personal/jrossi/cl_work/backrefs.csv')
    parser_opinion.add_argument('--csvpath', type=str, default='/ivi/ilps/personal/jrossi/cl_work/00_csv')
    parser_opinion.add_argument('--htmlpath', type=str, default='.')
    parser_opinion.set_defaults(func=extract_opinion)

    parser_tokens = subparsers.add_parser('tokens')
    parser_tokens.add_argument('--gist', type=str, help='JSON file with the gists')
    parser_tokens.add_argument('--cited_opid', nargs='+', type=int, help='Get the gist corresponding to the citation '
                                                                         'of those opinion_id(s)')
    parser_tokens.add_argument('--min', type=int, help='Get all opinions cited more than <min> times')
    parser_tokens.add_argument('--max', type=int, default=100000, help='Don\'t take opinions cited more '
                                                                       'than <max> times')
    parser_tokens.add_argument('--cache_tokens', type=str, help='Cache file with tokenized texts')
    parser_tokens.add_argument('--max_seq_len', type=int, default=32, help='Max Sequence length for text')
    parser_tokens.add_argument('--nb_workers', type=int, default=4)
    parser_tokens.set_defaults(func=tokens)

    parser_vecs = subparsers.add_parser('vecs')
    parser_vecs.add_argument('--cache_tokens', type=str, help='File with cached tokens. '
                                                              'Generated by the tokens command')
    parser_vecs.add_argument('--batch_size', type=int, help='Batch size for inference')
    parser_vecs.add_argument('--features', type=str, help='Path of the folder where the features will be stored')
    parser_vecs.set_defaults(func=vecs)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()


if __name__ == '__main__':
    main()
