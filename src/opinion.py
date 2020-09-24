import bs4
import json
import copy
import re

from typing import Iterator, Any, Dict, Optional
from bs4 import BeautifulSoup


DOCCANO_CITATION_TAG = 'CITATION'


class Opinion:
    """
    A helper class to manipulate an opinion.
    """
    bs4_citation_args = {'class': 'citation', 'data-id': True}

    def __init__(self, opinion_id: int, opinion_html: str):
        self.opinion_id = opinion_id
        self.opinion_html = opinion_html
        self.soup = clean_html(BeautifulSoup(self.opinion_html, 'html.parser'))
        self.raw_text = self.soup.get_text()
        self.num_words = len(self.raw_text.split())

        # MARK = replace all citations with easily findable strings in the raw text
        self.marked_soup = copy.copy(self.soup)
        for c in self.marked_soup.find_all('span', **Opinion.bs4_citation_args):
            c.string = f'MARK_FOR_CITATION_{c["data-id"]}'
        self.marked_text = clean_str(self.marked_soup.get_text())
        self.mark_regex = re.compile(r'MARK_FOR_CITATION_\w+')

    def citations(self, return_tag: bool = False) -> Iterator[Dict[str, Any]]:
        """
        An iterator to go through all the citations in the opinion. Each iteration will yield a dict with the keys
        citing_opinion_id, cited_opinion_id. If return_tag is True, then the dict also contains the key
        'tag', the value is a Tag object.
        :return: iterator
        """
        spans = self.soup.find_all('span', **Opinion.bs4_citation_args)
        for s in spans:
            data = {'citing_opinion_id': self.opinion_id, 'cited_opinion_id': int(s['data-id'])}
            if return_tag:
                data['tag'] = s
            yield data

    def citation_marks(self) -> Iterator[Dict[str, Any]]:
        """
        An iterator to go through the marked text. The marked text of an opinion is the full text where each citation
        is marked 'MARK_FOR_CITATION_<cited_opinion_id>'.
        The iterator yields dicts objects with keys 'marked_text' (value is a string), and 'span' (value is a tuple
        (start: int, end: int).
        :return: iterator
        """
        for mark in self.mark_regex.finditer(self.marked_text):
            mark: re.Match
            start, end = mark.span()
            yield {
                'marked_text': self.marked_text,  # memory-safe, no copy is made.
                'span': (start, end)
            }

    def __len__(self):
        return self.num_words


# Remove all \f \t (indentations)
_chars_to_clean = str.maketrans('', '', '\f\t')

# Regex substitutions: regex: what to substitute
_subs = {
    r'\u00ad\n *': '',
    r'\n +': '\n',
    r'\n{2,}': '\n',
    r' {2,}': ' ',
}


def clean_str(s: str) -> str:
    """
    Some cleaning of the raw text.
    Reduce to single newlines, try to put together word breaks.

    :param s: string to clean
    :return:
    """
    txt = s
    txt = txt.translate(_chars_to_clean)

    for regex, sub in _subs.items():
        txt = re.sub(regex, sub, txt)  # remove word break

    return txt


def clean_html(html: BeautifulSoup) -> BeautifulSoup:
    # A bit of cleaning on the original HTML
    # It includes tags for original pagination that insert numbers here and there in the text
    bs4_pagination_args = {'class': 'pagination'}
    bs4_citation_args_nolink = {'class': 'citation no-link'}

    for page in html.find_all('span', **bs4_pagination_args):
        page: bs4.element.Tag
        page.decompose()

    for nolink in html.find_all('span', **bs4_citation_args_nolink):
        nolink: bs4.element.Tag
        nolink.decompose()

    return html


def generate_doccano(opinion: Opinion, max_words_before_after: Optional[int] = None) -> Iterator[str]:
    """
    For each citation in an opinion, generate the snippet of text that will be used by annotators.
    It is returned in JSONL format, and the citation itself is marked as 'CITATION'.

    :param opinion: the opinion to process
    :param max_words_before_after: number of words of the text before and after the tag to include for the
    annotator. If None, then the whole text is returned
    :return: Iterator that yields JSONL strings
    """
    for citation in opinion.citation_marks():
        start, end = citation['span']
        marked_text = citation['marked_text']

        before_citation_txt = marked_text[:start]
        citation_txt = marked_text[start:end]
        after_citation_txt = marked_text[end:]

        if max_words_before_after is not None:
            before_citation_txt = ' '.join(before_citation_txt.split(' ')[-max_words_before_after:])
            after_citation_txt = ' '.join(after_citation_txt.split(' ')[:max_words_before_after])

        full_txt = ''.join([before_citation_txt, citation_txt, after_citation_txt])
        start_citation = len(before_citation_txt)
        end_citation = start_citation + len(citation_txt)

        yield json.dumps({'text': full_txt, 'labels': [[start_citation, end_citation, DOCCANO_CITATION_TAG]]})
