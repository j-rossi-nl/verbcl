import pandas as pd
import rouge
import sys
import tqdm
import textdistance
import re

from argparse import ArgumentParser
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from utils import df_from_file

# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args = None


def rouge_scores():
    """
    Create a dataframe with the ROUGE scores that compare REFERENCES texts to HYPOTHESIS texts
    :return:
    """
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'])

    refs: pd.DataFrame = df_from_file(_args.refs).dropna()
    hyps: pd.DataFrame = df_from_file(_args.hyps).dropna()

    uuids = refs[_args.refsuuid].unique()
    scores = []
    # UUID is very often the 'opinion_id' field. For each opinion, we compare the gists (the reference summary) to
    # the textrank summary (the hypothesis summary) to observe how the textrank captures the gist
    # Each score is a dictionary: {'metric1': {'r': 0.000, 'p': 0.000, 'f': 000}, 'metric2': {...} }
    for uuid in tqdm.tqdm(uuids):
        ref_txts = refs[refs[_args.refsuuid] == uuid][_args.refstext]
        hyp_txts = hyps[hyps[_args.hypsuuid] == uuid][_args.hypstext]

        # Make sure we actually have texts
        if len(ref_txts) * len(hyp_txts) == 0:
            continue

        score_dict = evaluator.get_scores([x for x in hyp_txts.values], [[x for x in ref_txts.values]])
        flat_score_dict = {'-'.join([k1, k2]): v for k1, sc in score_dict.items() for k2, v in sc.items()}
        flat_score_dict['uuid'] = uuid
        scores.append(flat_score_dict)

    # Create a Dataframe out of it
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(_args.dest, index=False)


def jaccard_scores():
    """
    Create a dataframe with the JACCARD SIMILARITY scores between 2 sets of texts
    :return:
    """
    left: pd.DataFrame = df_from_file(_args.left).dropna()
    right: pd.DataFrame = df_from_file(_args.right).dropna()

    def clean_txt_2_sequence(x):
        # Remove non-alpha / Lower case / Tokenize / Remove Stopwords / Stem
        letters_only = re.sub("[^a-zA-Z]",  # The pattern to search for
                              " ",  # The pattern to replace it with
                              x)  # The text to search
        lower_case = letters_only.lower()
        tokens = word_tokenize(lower_case)
        words = [w for w in tokens if w not in stopwords.words("english")]

        stemmer = PorterStemmer()
        stems = [stemmer.stem(word) for word in words]

        return stems

    def txts_2_sequence(txts):
        seqs = [clean_txt_2_sequence(x) for x in txts]
        return sum(seqs, [])

    # UUID is very often the 'opinion_id' field.
    # For each opinion we compare the JACCARD between texts from 2 sources
    uuids = left[_args.leftuuid].unique()
    scores = []

    for uuid in tqdm.tqdm(uuids):
        left_txts = left[left[_args.leftuuid] == uuid][_args.lefttext]
        right_txts = right[right[_args.rightuuid] == uuid][_args.righttext]

        # Make sure we actually have texts
        if len(left_txts) * len(right_txts) == 0:
            continue

        left_sequences = txts_2_sequence(left_txts)
        right_sequences = txts_2_sequence(right_txts)

        scores.append({'uuid': uuid, 'jaccard': textdistance.jaccard(left_sequences, right_sequences)})

    # Create a Dataframe out of it
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(_args.dest, index=False)


def parse_args(argstxt=None):
    if argstxt is None:
        argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    # Compute the ROUGE scores of summaries vs gists per court opinion
    # Expect to get 2 lists of summaries, one is the REFERENCE, the other the HYPOTHESIS
    # Uses the UUID ('opinion_id', ...) to identify summaries pertaining to the same opinion
    # This is NOT symetrical.
    parser_rouge = subparsers.add_parser('rouge')
    parser_rouge.add_argument('--refs', type=str, help='Path to the CSV/JSON with reference texts')
    parser_rouge.add_argument('--refstext', type=str, help='Column name for the TEXT in reference texts')
    parser_rouge.add_argument('--refsuuid', type=str, help='Column name for the UUID in reference texts')
    parser_rouge.add_argument('--hyps', type=str, help='Path to the CSV/JSON with hypothesis texts')
    parser_rouge.add_argument('--hypstext', type=str, help='Column name for the TEXT in hypothesis texts')
    parser_rouge.add_argument('--hypsuuid', type=str, help='Column name for the UUID in hypothesis texts')
    parser_rouge.add_argument('--dest', type=str, help='Path to the CSV with all the results')
    parser_rouge.set_defaults(func=rouge_scores)

    # Compute the JACCARD SIMILARITY scores of summaries vs gists per court opinion
    # Expect to get 2 lists of summaries LEFT and RIGHT
    # Uses the UUID ('opinion_id', ...) to identify summaries pertaining to the same opinion
    # This is symetrical
    parser_jaccard = subparsers.add_parser('jaccard')
    parser_jaccard.add_argument('--left', type=str, help='Path to the CSV/JSON with LEFT texts')
    parser_jaccard.add_argument('--lefttext', type=str, help='Column name for the TEXT in LEFT texts')
    parser_jaccard.add_argument('--leftuuid', type=str, help='Column name for the UUID in LEFT texts')
    parser_jaccard.add_argument('--right', type=str, help='Path to the CSV/JSON with RIGHT texts')
    parser_jaccard.add_argument('--righttext', type=str, help='Column name for the TEXT in RIGHT texts')
    parser_jaccard.add_argument('--rightuuid', type=str, help='Column name for the UUID in RIGHT texts')
    parser_jaccard.add_argument('--dest', type=str, help='Path to the CSV with all the results')
    parser_jaccard.set_defaults(func=jaccard_scores)

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
