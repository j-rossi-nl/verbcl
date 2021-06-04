import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd

from .opinion import Opinion

from typing import Iterator, Optional


class OpinionDataset:
    """
    A helper clas to handle a PARQUET dataset of opinions.
    By default, it will load the dataset into memory.
    """

    def __init__(self, path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Creates a dataset from a folder path. This folder contains multiple PARQUET files.
        :param path: dataset path
        """
        assert int(path is not None) + int(data is not None) == 1

        if path is not None:
            dataset = ds.dataset(path)
            self.df: pd.DataFrame = dataset.to_table().to_pandas().set_index('opinion_id')

        if data is not None:
            self.df = data

    def __getitem__(self, item: int) -> Opinion:
        """
        Get an opinion by its opinion_id

        :param item: has to be an integer
        :return: the opinion
        """
        if not isinstance(item, int):
            raise ValueError(f'The key should be an integer. Got {type(item)} instead.')

        # will raise KeyError if the item is not in the dataset
        data = self.df.loc[item]
        return Opinion(item, data['html_with_citations'])

    def __iter__(self):
        for op_id, op_html in self.df.iterrows():
            op_id: int
            yield Opinion(opinion_id=op_id, opinion_html=op_html['html_with_citations'])

    def __len__(self):
        return self.df.shape[0]


def opinions_in_arrowbatch(x: pa.RecordBatch) -> Iterator[Opinion]:
    """
    Helper to iterate through batches of opinion records coming from PARQUET dataset of opinions.

    :param x: batch
    :return: Opinion objects
    """
    d = x.to_pydict()
    for opinion_id, opinion_html in zip(d['opinion_id'], d['html_with_citations']):
        yield Opinion(opinion_id=opinion_id, opinion_html=opinion_html)
