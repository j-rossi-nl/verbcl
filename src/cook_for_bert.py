import glob
import os
import sys
import sentencepiece as spm
import json
import subprocess
import shlex

from nltk import RegexpTokenizer
from argparse import ArgumentParser

from utils import multiprocess_courtlistener

# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args = None
regex_tokenizer = RegexpTokenizer(r'\w+')


def normalize_text(text):
    # lowercase text
    text = str(text).lower()
    # remove non-UTF
    text = text.encode('utf-8', 'ignore').decode()
    # remove punktuation symbols
    text = " ".join(regex_tokenizer.tokenize(text))
    return text


def normalize_file(x):
    target_file_path = os.path.join(_args.dest, '{}_norm.txt'.format(os.path.basename(x)[:-4]))
    with open(x, encoding='utf-8') as fi:
        with open(target_file_path, encoding='utf-8', mode='w') as fo:
            for l in fi:
                fo.write('{}\n'.format(normalize_text(l)))
    return target_file_path


def normalize_all_texts():
    def normalize_all_texts_builder():
        return _args.txtpath, 'txt', _args.dest, 'txt', normalize_file
    multiprocess_courtlistener(process_builder=normalize_all_texts_builder,
                               nbworkers=_args.nbworkers)


def buildvocab():
    """
    Train a SentencePiece model based on the available text. Translates the vocabulary into a BERT vocabulary.
    Uses the text in the txt file of folder args.txtpath, the model will be named args.spmprefix, the vocabulary size
    is args.bertvocabsize. We use only a sample of the corpus, of size args.samplesize

    :return:
    """
    bert_config = json.load(open(_args.bertconfig))
    spm_command = (
        '--input={} '
        '--model_prefix={} '
        '--vocab_size={} '
        '--input_sentence_size={} '
        '--shuffle_input_sentence=true '
        '--bos_id=-1 --eos_id=-1'
    ).format(
        ','.join(glob.glob(os.path.join(_args.txtpath, '*.txt'))),
        _args.spmprefix,
        bert_config['vocab_size'] - bert_config['num_placeholders'],
        _args.samplesize
    )
    spm.SentencePieceTrainer.Train(spm_command)

    def read_sentencepiece_vocab(filepath):
        voc = []
        with open(filepath, encoding='utf-8') as fi:
            for line in fi:
                voc.append(line.split("\t")[0])
        # skip the first <unk> token
        voc = voc[1:]
        return voc

    # This creates 2 files in CWD : <prefix>.model and <prefix>.vocab
    # Move them to args.dest folder
    # model_filename = '{}.model'.format(args.spmprefix)
    vocab_filename = '{}.vocab'.format(_args.spmprefix)

    snt_vocab = read_sentencepiece_vocab(vocab_filename)

    # Now we turn the SentencePiece vocabulary into a BERT vocabulary
    def parse_sentencepiece_token(tk):
        if tk.startswith('▁'):
            return tk[1:]
        else:
            return "##" + tk

    bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))
    ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    bert_vocab = ctrl_symbols + bert_vocab
    bert_vocab += ["[UNUSED_{}]".format(i) for i in range(bert_config['vocab_size'] - len(bert_vocab))]
    with open(os.path.join(_args.dest, 'vocab.txt'), "w") as fo:
        for token in bert_vocab:
            fo.write(token + "\n")


def txt_to_tfrecord(x):
    """
    Generate the tfrecord files, based on the normalized TXT files generated by normalize, and the vocabulary
    generated by buildvocab
    :return:
    """
    bert_config = json.load(open(os.path.join(_args.bertconfigpath, 'bert_config.json')))
    xargs_cmd = ("python3 {} "
                 "--input_file={} "
                 "--output_file={} "
                 "--vocab_file={} "
                 "--do_lower_case={} "
                 "--max_predictions_per_seq={} "
                 "--max_seq_length={} "
                 "--masked_lm_prob={} "
                 "--random_seed=42 "
                 "--dupe_factor=5")

    run_cmd = xargs_cmd.format(
        os.path.join(_args.bertsrcpath, 'create_pretraining_data.py'),
        x,
        os.path.join(_args.pretrainpath, '{}.tfrecord'.format(os.path.basename(x)[:-4])),
        os.path.join(_args.bertconfigpath, 'vocab.txt'),
        bert_config['do_lower_case'],
        bert_config['max_predictions'],
        bert_config['max_position_embeddings'],
        bert_config['masked_lm_prob']
    )
    subprocess.run(shlex.split(run_cmd), capture_output=True)


def generate_pretraining_data():
    def generate_pretraining_data_builder():
        return _args.txtpath, 'txt', _args.pretrainpath, 'tfrecord', txt_to_tfrecord
    multiprocess_courtlistener(process_builder=generate_pretraining_data_builder, nbworkers=_args.nbworkers)


def split(x):
    xargs_cmd = (
        'split -a 5 '
        '-l {} '
        '-d '
        '--additional-suffix=.txt '
        '{} '
        '{}'
    )
    run_cmd = xargs_cmd.format(
        _args.size,
        x,
        os.path.join(_args.dest, '{}_'.format(os.path.basename(x)[:-4]))
    )
    print(run_cmd)
    subprocess.run(shlex.split(run_cmd), capture_output=True, check=True)


def split_files():
    def split_files_builder():
        return _args.txtpath, 'txt', _args.dest, 'txt', split
    multiprocess_courtlistener(process_builder=split_files_builder, nbworkers=_args.nbworkers, monitoring=False)


def parse_args(argstxt=sys.argv[1:]):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    parser_normalize = subparsers.add_parser('normalize')
    parser_normalize.add_argument('--txtpath', type=str, help='The path to the folder with all *.txt '
                                                              'files')
    parser_normalize.add_argument('--dest', type=str, help='Destination path for the txt file with normalized text')
    parser_normalize.add_argument('--nbworkers', type=int, default=4, help='Number of parallel workers')
    parser_normalize.set_defaults(func=normalize_all_texts)

    parser_split = subparsers.add_parser('split')
    parser_split.add_argument('--txtpath', type=str, help='Folder with original txt files')
    parser_split.add_argument('--dest', type=str, help='Destination folder for split files')
    parser_split.add_argument('--size', type=int, default='200000', help='Size of chunksin numer of lines')
    parser_split.add_argument('--nbworkers', type=int, help='Number of parallel workers')
    parser_split.set_defaults(func=split_files)

    parser_vocab = subparsers.add_parser('buildvocab')
    parser_vocab.add_argument('--txtpath', type=str, help='The path to the folder with all *.txt files')
    parser_vocab.add_argument('--spmprefix', type=str, help='Prefix for SPM model files')
    parser_vocab.add_argument('--samplesize', type=int, default=int(2e7), help='We only sample the corpus')
    parser_vocab.add_argument('--bertconfig', type=str, help='Configuration file for BERT model and pre-training')
    parser_vocab.add_argument('--dest', type=str, help='Path to copy all generated files')
    parser_vocab.add_argument('--nbworkers', type=int, default=4, help='Number of parallel workers')
    parser_vocab.set_defaults(func=buildvocab)

    parser_generate = subparsers.add_parser('generate')
    parser_generate.add_argument('--bertconfigpath', type=str, help='Path to config folder for BERT model '
                                                                    'and pre-training')
    parser_generate.add_argument('--bertsrcpath', type=str, help='Folder with the git clone of bert source code')
    parser_generate.add_argument('--txtpath', type=str, help='Folder with all txt files')
    parser_generate.add_argument('--pretrainpath', type=str, help='Path to pre-training material')
    parser_generate.add_argument('--nbworkers', type=int, default=4, help='Number of parallel workers')
    parser_generate.set_defaults(func=generate_pretraining_data)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()


if __name__ == '__main__':
    try:
        main()
    except Exception as excp:
        print(excp)
        import pdb
        pdb.post_mortem()