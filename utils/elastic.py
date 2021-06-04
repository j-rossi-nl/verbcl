import elasticsearch
import logging
import os
import pandas as pd
import pyarrow.dataset as ds
import shutil

from dotenv import load_dotenv
from elasticsearch_dsl import analyzer, connections
from elasticsearch_dsl import Boolean, Document, Integer, Text
from typing import Any, List, Type

from .misc import batch_iterator, random_name
from .multiprocess import queue_worker, multiprocess

elasticsearch.logger.setLevel(logging.WARNING)


class OpinionDocument(Document):
    opinion_id = Integer()
    raw_text = Text(analyzer=analyzer('alpha_stop_stem',
                                      type='custom',
                                      tokenizer='classic',
                                      filter=['lowercase', 'asciifolding', 'stop', 'snowball']))

    class Index:
        name = 'juju-01'


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
        name = 'juju-02'

    # Overloading save() to set False as the default value for highlight
    def save(self, **kwargs):
        if self.highlight is None:
            self.highlight = False
        if self.count_citations is None:
            self.count_citations = 0

        return super().save(**kwargs)


def elastic_init(envfile: str):
    # Initialize the connection to Elasticsearch
    # Making use of elasticsearch_dsl persistence features
    load_dotenv(os.path.expanduser(envfile))
    if os.getenv('ELASTIC_CLOUD_ID') is not None:
        connections.create_connection(cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
                                      api_key=(
                                          os.getenv('ELASTIC_CLOUD_API_ID'), os.getenv('ELASTIC_CLOUD_API_KEY')),
                                      timeout=1000)
    else:
        connections.create_connection(host=os.getenv('ELASTIC_HOST'),
                                      port=os.getenv('ELASTIC_PORT'),
                                      timeout=1000,
                                      maxsize=256)

    OpinionDocument.init()
    OpinionSentence.init()


def gather_by_opinion_ids(class_: Type[Document], opinion_ids: List[int], envfile: str, nb_workers: int = 4,
                          batch_size: int = 32):
    # Use ElasticSearch to assemble all the sentences from the given opinion ids.
    tmpfolder = os.path.join("/tmp", f"juju_{random_name()}/")
    os.makedirs(tmpfolder)

    @queue_worker
    def _gather(opinion_ids_: List[int]) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        elastic_init(envfile)

        search = class_.search().query("terms", opinion_id=opinion_ids_)
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
    df = dataset.to_table().to_pandas()
    shutil.rmtree(tmpfolder)
    return df
