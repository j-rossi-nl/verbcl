import json
import re

from typing import Iterator
from functools import partial

from utils import text_before_after
from opinion import Opinion, CITATION_MARK


DOCCANO_CITATION_TAG = 'CITATION'


def generate_doccano(opinion: Opinion, max_words_before_after: int) -> Iterator[str]:
    """
    For each citation in an opinion, generate the snippet of text that will be used by annotators.
    It is returned in JSONL format, and the citation itself is marked as 'CITATION'.

    :param opinion: the opinion to process
    :param max_words_before_after: number of words of the text before and after the tag to include for the
    annotator. If None, then the whole text is returned
    :return: Iterator that yields JSONL strings
    """
    for citation in opinion.citation_marks():
        full_txt, len_before, len_after = text_before_after(citation['marked_text'],
                                                            citation['span'],
                                                            max_words_before_after)
        len_span = citation['span'][1] - citation['span'][0]

        # Remove the other CITATION marks out of the current citation.
        # It happens regularly that other citations are nearby, this could confuse the annotator
        before_citation_txt = full_txt[:len_before]
        after_citation_txt = full_txt[-len_after:]
        before_citation_txt, after_citation_txt = list(map(partial(re.sub, CITATION_MARK, f'OTHER_{CITATION_MARK}'),
                                                           [before_citation_txt, after_citation_txt]))

        full_txt = ''.join([before_citation_txt,
                            full_txt[len_before:len_before + len_span],
                            after_citation_txt])
        start_citation = len(before_citation_txt)
        end_citation = start_citation + len_span

        yield json.dumps({'text': full_txt, 'labels': [[start_citation, end_citation, DOCCANO_CITATION_TAG]]})
