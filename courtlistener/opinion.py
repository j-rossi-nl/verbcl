import bs4
import copy
import re
import json

from functools import partial
from typing import Iterator, Any, Dict, Tuple
from bs4 import BeautifulSoup


CITATION_MARK = 'CITATION_@@_'
DOCCANO_CITATION_TAG = 'CITATION'


class Opinion:
    """
    A helper class to manipulate an opinion.
    """
    bs4_citation_args = {'class': 'citation', 'data-id': True}
    mark_regex = re.compile(f'{CITATION_MARK}' + r'\w+')
    verbatim_quote = re.compile(r'"(?P<quote>.+?)"')

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

    def doccano(self, max_words_before_after: int) -> Iterator[str]:
        """
        For each citation in an opinion, generate the snippet of text that will be used by annotators.
        It is returned in JSONL format, and the citation itself is marked as 'CITATION'.

        :param max_words_before_after: number of words of the text before and after the tag to include for the
        annotator. If None, then the whole text is returned
        :return: Iterator that yields JSONL strings
        """
        for citation in self.citation_marks():
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

    def verbatim(self, max_words_before_after: int, min_words_verbatim: int) -> Iterator[Dict[str, Any]]:
        """
        An iterator that yields potential verbatim quotes from cited opinions.
        Warning: it does not check that the verbatim quotes are indeed from the cited opinion

        :return: an iterator
        """
        for citation in self.citations_and_marks():
            snippet_txt, len_before, len_after = text_before_after(citation['marked_text'],
                                                                   citation['span'],
                                                                   nb_words=max_words_before_after)

            # Find all verbatim quote in the text before the citation
            for m in filter(lambda x: len(x['quote'].split()) >= min_words_verbatim,
                            self.verbatim_quote.finditer(snippet_txt[:-len_after])):
                yield {
                    'citing_opinion_id': self.opinion_id,
                    'cited_opinion_id': citation['cited_opinion_id'],
                    'verbatim': m['quote'],
                    'snippet': snippet_txt,
                    'span_in_snippet': m.span()
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
        txt = re.sub(regex, sub, txt)

    return txt


def clean_html(html: BeautifulSoup) -> BeautifulSoup:
    # A bit of cleaning on the original HTML
    # It includes tags for original pagination that insert numbers here and there in the text
    decompose = [
        'pagination',
        'citation no-link',
        'star-pagination',
    ]

    for d in decompose:
        bs4_args = {'class': d}
        for f in html.find_all('span', **bs4_args):
            f: bs4.element.Tag
            f.decompose()

    return html


def text_before_after(txt: str, span: Tuple[int, int], nb_words: int) -> Tuple[str, int, int]:
    """
    Given a text, and span within this text, extract a number of words before and after the span.
    The returned text includes the span within the original text, surrounded by nb_words.

    :param txt: original text
    :param span: a 2-uple (start, end) indicating the span of text to be preserved
    :param nb_words: how many words to extract
    :return: a snippet of text, length of text before the original span, length of text after the original span
    """
    start, end = span
    before_txt = txt[:start]
    span_txt = txt[start:end]
    after_txt = txt[end:]

    before_txt = ' '.join(before_txt.split(' ')[-nb_words:])
    after_txt = ' '.join(after_txt.split(' ')[:nb_words])

    total_txt = ''.join([before_txt, span_txt, after_txt])
    return total_txt, len(before_txt), len(after_txt)
