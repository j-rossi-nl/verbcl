import re
import bs4
import en_core_web_sm

from spacy.lang.en import English
from spacy.tokens import Doc
from bs4 import BeautifulSoup
from typing import List, Dict, Callable, Optional, Any

from nlp_nonsense import sentence_or_not_spacy

LAST_WORDS_EXTRACT = 20

# The SpaCy NLP engine
_nlp = en_core_web_sm.load()
_nlp: English
_nlp.add_pipe(_nlp.create_pipe('sentencizer'))


# Limit to N words when reading the text backwards
LIMIT_NB_WORDS = 100
_quote_in_sentence = re.compile(r'^.*"(?P<quote>[^"]+)".+')


def _nlp_extract(txt: str) -> Optional[str]:
    """
    Extract the supposed catchphrase by using Spacy NLP.
    The target is to find fully formed sentence the closest to the end of the text (we assume the text
    ends right before the citation link.
    :param txt:
    :return:
    """
    # As a heuristic, we consider only the last 5 sentences
    doc: Doc = next(_nlp.pipe([txt], disable=['parser']))
    candidate_sentences = list(doc.sents)[-5:]
    acceptable_sentences = sentence_or_not_spacy(candidate_sentences)
    if any(acceptable_sentences):
        # The first sentence from the end of the list (so the closest to the citation)
        # that is considered to be a valid sentence
        index = len(acceptable_sentences) - acceptable_sentences[::-1].index(True) - 1
        candidate_sentence = str(candidate_sentences[index])
    else:
        return None

    # Now in the selected text before the citation, we retain either the last full sentence, or a text
    # text enclosed in quotemarks in the last full sentence
    # For example: J could ask for ... against ... as in G v R <citation> --> full sentence
    # Or: J could ask ... as "... ... ..." in G vR <citation>             --> only what's in quotes
    m = _quote_in_sentence.match(candidate_sentence)
    op_gist = m.group('quote') if m is not None else candidate_sentence.strip()
    return op_gist


def _last_extract(txt: str) -> str:
    op_gist = ' '.join(txt.split()[-LAST_WORDS_EXTRACT:])
    return op_gist


_methods_fn: Dict[str, Callable[[str], Optional[str]]] = {
    'nlp': _nlp_extract,
    'last': _last_extract
}


def clean_html(html: BeautifulSoup) -> BeautifulSoup:
    # A bit of cleaning on the original HTML
    # It includes tags for original pagination that insert numbers here and there in the text
    bs4_pagination_args = {'class': 'pagination'}
    bs4_nolink_citation_args = {'class': 'citation no-link'}

    for page in html.find_all('span', **bs4_pagination_args):
        page: bs4.element.Tag
        page.decompose()

    for nolink in html.find_all('span', **bs4_nolink_citation_args):
        nolink: bs4.element.Tag
        nolink.decompose()

    return html


def extract_catchphrase(html_with_citations: str,
                        method: str) -> List[Dict[str, Any]]:
    """
    Extract for each citation, the 'GIST' of it, which is the catchphrase that introduces the citED opinion
    in the citING opinion. Each cited opinion is introduced by a sentence that underlines what from the cited
    opinion is an argument for the citing opinion.

    :param html_with_citations: original HTML of the court opinion
    :param method: a method like 'nlp' or 'last'
    :return:
    """
    bs4_citation_args = {'class': 'citation', 'data-id': True}
    html = BeautifulSoup(html_with_citations, 'html.parser')

    processed_ids = []
    opinion_gists = []
    for cited in html.find_all('span', **bs4_citation_args):
        cited: bs4.element.Tag

        # One citation can generate multiple spans in the document.
        # Guess it is an artefact of the html generation.
        # We keep track of which ones have been processed.
        # TODO : it means that we restrict to the first time an opinion is cited, by interpreting multiple
        # TODO : citations as an incorrect HTML generation
        cited_opinion_id = cited['data-id']
        if cited_opinion_id in processed_ids:
            continue
        processed_ids.append(cited_opinion_id)

        # The catchphrase will be before the citation in the text, close by, and will be a full sentence
        # As a heuristic, we go through the previous siblings in the HTML parsing and retain the first one
        # We assemble the text located before the citation, but still within the same parent in the HTML
        # as a proxy to the text of the paragraph where the citation is
        # We limit to N former words
        cumulated_txt = ''
        for t in cited.previous_siblings:
            txt: str = t.text if isinstance(t, bs4.element.Tag) else t
            txt = txt.replace('\n', ' ')
            cumulated_txt = '{} {}'.format(txt, cumulated_txt)
            nb_words_cumulated = len(cumulated_txt.split())
            if nb_words_cumulated > LIMIT_NB_WORDS:
                break

        # If no text, no further effort
        if len(cumulated_txt) == 0:
            continue

        _call = _methods_fn[method]
        op_gist = _call(cumulated_txt)
        if op_gist is not None:
            opinion_gists.append({'cited_opinion_id': cited_opinion_id, 'gist': op_gist})

    return opinion_gists
