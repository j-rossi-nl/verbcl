import bs4
import copy
import re

from typing import Iterator, Any, Dict
from bs4 import BeautifulSoup

CITATION_MARK = 'CITATION_@@_'


class Opinion:
    """
    A helper class to manipulate an opinion.
    """
    bs4_citation_args = {'class': 'citation', 'data-id': True}
    mark_regex = re.compile(f'{CITATION_MARK}' + r'\w+')

    def __init__(self, opinion_id: int, opinion_html: str):
        self.opinion_id = opinion_id
        self.opinion_html = opinion_html
        self.soup = clean_html(BeautifulSoup(self.opinion_html, 'html.parser'))
        self.raw_text = self.soup.get_text()
        self.num_words = len(self.raw_text.split())

        # MARK = replace all citations with easily findable strings in the raw text
        self.marked_soup = copy.copy(self.soup)
        for c in self.marked_soup.find_all('span', **Opinion.bs4_citation_args):
            c.string = f'{CITATION_MARK}{c["data-id"]}'
        self.marked_text = clean_str(self.marked_soup.get_text())

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

    def citations_and_marks(self) -> Iterator[Dict[str, Any]]:
        """
        An iterator that yields the data from citation_marks() and citations()
        :return: an iterator
        """
        for citation, mark in zip(self.citations(), self.citation_marks()):
            yield {**citation, **mark}

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
        txt = re.sub(regex, sub, txt)

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
