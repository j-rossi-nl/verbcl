import logging
import gensim

from gensim.summarization.summarizer import summarize
from typing import Dict, Callable


gensim_logger = logging.getLogger(gensim.__name__)
gensim_logger.setLevel(logging.ERROR)


def textrank(txt):
    return summarize(txt)


summarization_methods: Dict[str, Callable[[str], str]] = {
    'textrank': textrank
}
