# noinspection PyUnresolvedReferences
import pytextrank
import spacy

from nltk.tokenize import sent_tokenize
from spacy.tokens import Doc
from typing import Callable, Dict, List, Union


class CustomNLP:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")
        self.nlp.max_length = 4e6

    def __call__(self, text: Union[str, List[str]]) -> Doc:
        """
        From a raw text, creates a spacy Document, using NLTK as tokenizer and sentence tokenizer.
        :param text:
        :return:
        """
        if isinstance(text, str):
            nltk_sents = sent_tokenize(text)
        elif isinstance(text, list) and all(isinstance(x, str) for x in text):
            nltk_sents = text
        else:
            raise ValueError(f"text should be a string or a list of strings.")

        return self._from_sentences(nltk_sents)

    def _from_sentences(self, sents: List[str]) -> Doc:
        """
        From a list of sentences, form a single document.

        :param sents:
        :return:
        """
        docs: List[Doc] = []
        for s in sents:
            doc = self.nlp(s, disable=["parser", "textrank"])
            doc[0].is_sent_start = True
            for t in doc[1:]:
                t.is_sent_start = False
            docs.append(doc)

        # Concatenate the sentences in one Doc
        # noinspection PyTypeChecker
        doc = Doc.from_docs(docs=docs)

        # Run the rest of the pipeline (the tagger and parser will overwrite neither the tokenization
        # nor the sentence tokenization
        for name, proc in self.nlp.pipeline:  # Iterate over components in order
            doc = proc(doc)

        return doc


def summarizer(method: str, **kwargs) -> Callable[[str], str]:
    """
    Return a function that takes IN a text and returns OUT a summary, based on the arguments provided in kwargs.

    :param method:
    :param kwargs: "nb_sentences": the number of sentences to include in the summary
    :return: a callable func(str) -> str
    """
    nlp = CustomNLP()

    def _make_textrank(**kwargs):
        def _textrank(txt):
            doc = nlp(txt)
            tr = doc._.textrank
            summary = ''.join(s.text for s in tr.summary(limit_sentences=kwargs["nb_sentences"]))
            return summary
        return _textrank

    method_to_func: Dict[str, Callable[[...], Callable[[str], str]]] = {
        "textrank": _make_textrank
    }

    return method_to_func[method](**kwargs)
