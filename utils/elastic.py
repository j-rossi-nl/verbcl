import elasticsearch
import logging
import multiprocessing
import os
import pandas as pd
import pyarrow.dataset as ds
import shutil
import tqdm

from dotenv import load_dotenv
from elasticsearch_dsl import analyzer, connections
from elasticsearch_dsl import Boolean, Document, Float, Integer, Text
from typing import Any, List, Type

from .misc import batch_iterator, make_clean_folder, random_name
from .multiprocess import queue_worker, multiprocess

elasticsearch.logger.setLevel(logging.WARNING)


class OpinionDocument(Document):
    opinion_id = Integer()
    raw_text = Text(analyzer=analyzer('alpha_stop_stem',
                                      type='custom',
                                      tokenizer='classic',
                                      filter=['lowercase', 'asciifolding', 'stop', 'snowball']))

    class Index:
        name = 'verbcl_opinions'


class OpinionSentence(Document):
    opinion_id = Integer()
    sentence_id = Integer()
    highlight = Boolean()
    count_citations = Integer()
    raw_text = Text(analyzer=analyzer('alpha_stop_stem',
                                      type='custom',
                                      tokenizer='classic',
                                      filter=['lowercase', 'asciifolding', 'stop', 'snowball']))

    class Index:
        name = 'verbcl_highlights'

    # Overloading save() to set False as the default value for highlight
    def save(self, **kwargs):
        if self.highlight is None:
            self.highlight = False
        if self.count_citations is None:
            self.count_citations = 0

        return super().save(**kwargs)


class OpinionCitationGraph(Document):
    citing_opinion_id = Integer()
    cited_opinion_id = Integer()
    cited_sentence_id = Integer()
    verbatim = Text(analyzer=analyzer('alpha_stop_stem',
                                      type='custom',
                                      tokenizer='classic',
                                      filter=['lowercase', 'asciifolding', 'stop', 'snowball']))
    snippet = Text(analyzer=analyzer('alpha_stop_stem',
                                     type='custom',
                                     tokenizer='classic',
                                     filter=['lowercase', 'asciifolding', 'stop', 'snowball']))
    score = Float()

    class Index:
        name = 'verbcl_citation_graph'


def _create_connection(alias_name: str):
    if os.getenv('ELASTIC_CLOUD_ID') is not None:
        connections.create_connection(alias=alias_name,
                                      cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
                                      api_key=(
                                          os.getenv('ELASTIC_CLOUD_API_ID'), os.getenv('ELASTIC_CLOUD_API_KEY')),
                                      timeout=1000)
    else:
        connections.create_connection(alias=alias_name,
                                      host=os.getenv('ELASTIC_HOST'),
                                      port=os.getenv('ELASTIC_PORT'),
                                      timeout=1000,
                                      maxsize=256)


# Global state variables for the connection management
_read_env_file: bool = False
_class_init: bool = False
_list_aliases: List[str] = []
_default_connection: str = "default"


def elastic_init(envfile: str) -> str:
    """
    Manages connections to ElasticSearch instance.
    There is one connection per process.

    :param envfile:
    :return: alias name for the connection to use
    """
    # Initialize the connection to Elasticsearch
    # Making use of elasticsearch_dsl persistence features
    global _read_env_file
    if not _read_env_file:
        load_dotenv(os.path.expanduser(envfile))
        _read_env_file = True

    proc = multiprocessing.current_process()
    alias_name = proc.name

    global _list_aliases
    if alias_name in _list_aliases:
        return alias_name

    _create_connection(alias_name)
    _list_aliases.append(alias_name)


    global _class_init
    if not _class_init:
        # Always have a connection named "default"
        if _default_connection not in _list_aliases:
            _create_connection(_default_connection)

        OpinionDocument.init()
        OpinionSentence.init()
        OpinionCitationGraph.init()
        _class_init = True

    return alias_name


def gather_by_opinion_ids(class_: Type[Document], opinion_ids: List[int], envfile: str, nb_workers: int = 4,
                          batch_size: int = 32) -> pd.DataFrame:
    # Use ElasticSearch to assemble all the sentences from the given opinion ids.
    tmpfolder = os.path.join("/tmp", f"juju_{random_name()}/")
    os.makedirs(tmpfolder)

    @queue_worker
    def _gather(opinion_ids_: List[int]) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        alias = elastic_init(envfile)

        search = class_.search(using=alias).query("terms", opinion_id=opinion_ids_)
        result = search.scan()
        data = [r.to_dict() for r in result]
        pd.DataFrame(data).to_parquet(os.path.join(tmpfolder, f"{random_name(16)}.parq"))
        return len(opinion_ids_)

    iterator, nb_rows = batch_iterator(items=opinion_ids, batch_size=batch_size)
    multiprocess(worker_fn=_gather, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=nb_workers, description=f'Get data from Elastic.')

    # Gather all data, Clean TMP folder
    dataset: Any = ds.dataset(tmpfolder)
    dataset: ds.FileSystemDataset
    df: pd.DataFrame = dataset.to_table().to_pandas()
    shutil.rmtree(tmpfolder)
    return df


def collection_to_parquet(class_: Type[Document], envfile: str, destination: str, batch_size: int = 1000000):
    # Use ElasticSearch to assemble all the sentences from the given opinion ids.
    make_clean_folder(destination)

    # Search
    alias = elastic_init(envfile)

    search = class_.search(using=alias)
    count = search.count()
    results = search.scan()

    for _ in tqdm.tqdm(range(0, count, batch_size)):
        data = [r.to_dict() for _, r in tqdm.tqdm(zip(range(batch_size), results), total=batch_size)]
        pd.DataFrame(data).to_parquet(os.path.join(destination, f"{random_name(16)}.parq"))
