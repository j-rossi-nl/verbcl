import regex
import re

from typing import Iterator, Dict, Any
from opinion import Opinion
from utils import text_before_after
from opinion_dataset import OpinionDataset


verbatim_quote = re.compile(r'"(?P<quote>.+)"')


def verbatim(citing_opinion: Opinion, search_radius: int, cited_opinions: OpinionDataset) -> Iterator[Dict[str, Any]]:
    """
    Extracts verbatim quotes from the cited opinion.
    Heuristic: look for " " before the citation in the text. A verbatim quote is a citation

    :param citing_opinion: a snippet of text containing a citation
    :param search_radius: how many words before/after the citation should be considered
    :param cited_opinions: a dataset in which we can find the cited opinions
    :return: Iterator that yields a dictionary
    """
    for citation in citing_opinion.citations_and_marks():
        snippet_txt, len_before, len_after = text_before_after(citation['marked_text'],
                                                               citation['span'],
                                                               search_radius)

        # 1) Find a verbatim quote in the text before the citation
        m = verbatim_quote.search(snippet_txt)
        if m is None:
            # No quote located in the text
            continue

        # 2) If there is a verbatim quote, confirm that it comes from the cited opinion
        verbatim_txt = m['quote']
        try:
            cited = cited_opinions[citation['cited_opinion_id']]
        except KeyError:
            continue

        cited_full_txt = cited.raw_text

        try:
            quote_match = regex.search(f'({verbatim_txt})' + r'{e<10}', cited_full_txt, flags=regex.IGNORECASE)
        except regex.error:
            continue

        if quote_match is None:
            # The located quote is not from the cited opinion
            continue

        # 3) Yield the relevant data
        yield {
            'citing_opinion_id': citing_opinion.opinion_id,
            'span_in_citing_marked_text': m.span(),
            'cited_opinion_id': citation['cited_opinion_id'],
            'span_in_cited_raw_text': quote_match.span(),
        }
