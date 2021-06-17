import elasticsearch
import logging
import numpy as np
import os
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.dataset as ds
import sys
import tqdm

from argparse import ArgumentParser, Namespace
from bs4 import BeautifulSoup
from collections import defaultdict
from elasticsearch import RequestError
from elasticsearch_dsl import Search
from nltk.tokenize import sent_tokenize
from socket import gethostname
from threading import Thread
from typing import Any, Dict, List, Tuple

from courtlistener import opinions_in_arrowbatch
from utils import OpinionCitationGraph, OpinionDocument, OpinionSentence
from utils import batch_iterator, collection_to_parquet, config, elastic_init, \
    queue_worker, make_clean_folder, multiprocess, parquet_dataset_iterator, random_name, \
    read_jsonl, write_jsonl

config()
elasticsearch.logger.setLevel(logging.WARNING)
_args: Namespace = Namespace()

# Deeper recursion for BeautifulSoup
max_recursion = 100000
sys.setrecursionlimit(max_recursion)


def index_all_opinions():
    """
    Adds all opinions to an index.

    :return:
    """

    @queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        alias = elastic_init(_args.env)

        for opinion in opinions_in_arrowbatch(x):
            OpinionDocument(opinion_id=opinion.opinion_id,
                            raw_text=opinion.raw_text).save(using=alias)
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=8)
    multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description=f'Add opinions to index {OpinionDocument.Index.name}')


def index_all_sentences():
    """
    Adds all sentences of each opinion to an index.

    :return:
    """

    @queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        alias = elastic_init(_args.env)

        for opinion in opinions_in_arrowbatch(x):
            for sentence_id, sentence in enumerate(sent_tokenize(opinion.raw_text)):
                try:
                    OpinionSentence(
                        opinion_id=opinion.opinion_id,
                        sentence_id=sentence_id,
                        raw_text=sentence,
                        highlight=False,
                        count_citations=0
                    ).save(using=alias)
                except RequestError as e:
                    logging.debug(f"opinion_id={opinion.opinion_id}, sentence_id={sentence_id}, "
                                  f"raw_text={sentence}, Error={repr(e)}")
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=8)
    multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description=f'Add opinions to index {OpinionSentence.Index.name}')


def index_all_citation_graph():
    """
    From the PARQUET dataset of the citation graph to ElasticSearch.
    :return:
    """

    @queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        alias = elastic_init(_args.env)

        df: pd.DataFrame = x.to_pandas()
        for _, row in df.iterrows():
            OpinionCitationGraph(
                citing_opinion_id=row['citing_opinion_id'],
                cited_opinion_id=row['cited_opinion_id'],
                verbatim=row['verbatim'],
                snippet=row['snippet'],
                score=row['score']
            ).save(using=alias)

        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=256)
    multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers,
                 description=f'Add citation edges to index {OpinionCitationGraph.Index.name}')


def _create_search_verbatim(cited_opinion_id: int, full: str) -> Search:
    first_10 = ' '.join(full.split(' ')[:10])
    last_10 = ' '.join(full.split(' ')[-10:])

    # Using INTERVALS query
    # Looking for the whole verbatim, the first 10 words, the last 10 words
    raw_text_query = {
        'any_of': {
            'intervals': [
                {'match': {'query': full, 'ordered': True, 'max_gaps': 10}},
                {'match': {'query': first_10, 'ordered': True, 'max_gaps': 2}},
                {'match': {'query': last_10, 'ordered': True, 'max_gaps': 2}}
            ]
        }
    }

    # Which connection to use?
    alias = elastic_init(_args.env)

    return OpinionDocument.search(using=alias). \
        query("match", opinion_id=cited_opinion_id). \
        query("intervals", raw_text=raw_text_query)


def search_verbatims():
    """
    From all the potential verbatim quotes, identify the true citations that were made through a verbatim quote
    Using Elastic Search with the quote.

    :return:
    """

    @queue_worker
    def _search(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features

        data = []
        batch: pd.DataFrame = x.to_pandas()
        for _, d in batch.iterrows():
            d: pd.Series
            cited_opinion_id = d['cited_opinion_id']
            full = d['verbatim']

            # STEP 1 - YES or NO: is the verbatim from the cited opinion ??
            base_search = _create_search_verbatim(cited_opinion_id, full)
            score_search = base_search.params(filter_path=['hits.hits._score'])

            try:
                results = score_search.execute()
                score = results[0].meta.score
            except (RequestError, KeyError, IndexError):
                # The search did not return any result at all
                # The PARAMS restrain the query to ONE document, and the INTERVALS queries will dismiss
                # documents that do not match the rules.
                # So this happens when the alledged verbatim does not appear in the cited opinion
                score = -1
            except Exception:
                score = -1

            data.append({**(d.to_dict()), 'score': float(score)})
            if score == -1:
                continue

        # Save the Dataset for YES/NO as PARQ
        if _args.save and len(data) > 0:
            file_out = os.path.join(_args.dest, f'{random_name()}.parq')
            pd.DataFrame(data).to_parquet(file_out)

        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    logging.info(f'Destination {_args.dest}')
    make_clean_folder(_args.dest)
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=128)
    multiprocess(worker_fn=_search, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description='Search Verbatims')


def search_verbatim_sentence():
    """
    From all the verbatim citations, identify which sentence was cited from the cited opinion.
    This does not generate any dataset, it updated the ElasticSearch index of OpinionSentence.

    :return:
    """
    # Global variables that will be updated by processes and threads
    tmpfile_sentence = _args.tmp_sentence
    if tmpfile_sentence is None:
        tmpfile_sentence = os.path.join("/tmp", f"juju_{random_name()}")

    tmpfile_citation = _args.tmp_citation
    if _args.tmp_citation is None:
        tmpfile_citation = os.path.join("/tmp", f"juju_{random_name()}")

    logging.info(f"Running on {gethostname()}")
    logging.info(f"Using {tmpfile_sentence} to store OpinionSentence updates.")
    logging.info(f"Using {tmpfile_citation} to store OpinionCitationGraph updates.")

    def _search_sentence(cited_opinion_id: int, verbatim: str):
        # Based on the same search as above
        base_search = _create_search_verbatim(cited_opinion_id, verbatim)

        # Use HIGHLIGHT with the previous search to identify WHERE the matching occurs
        verbatim_search = base_search. \
            highlight("raw_text",
                      type="unified",
                      fragment_size=1000,
                      order="score",
                      number_of_fragments=1). \
            params(filter_path=['hits.hits.highlight'])

        results = verbatim_search.execute()
        highlight = results[0].meta['highlight']['raw_text'][0]

        # The highlight with highest score is returned
        # It is a text with more than one sentence
        # We select the sentence that includes the highest number of highlighted terms (<em> in Highlight HTML)
        highlight_sentences = map(lambda s: s if "<em>" in s else "", sent_tokenize(highlight))
        soups = [BeautifulSoup(s, "html.parser") for s in highlight_sentences]

        count_em = [len(s.find_all("em")) for s in soups]
        best_index = np.argmax(count_em)
        best_text = soups[best_index].text

        # Use the database of opinion sentences to find the ID of this sentence in the cited opinion
        sentence_query = {
            "all_of": {
                "intervals": [
                    {"match": {"query": best_text, "ordered": True, "max_gaps": 1}}
                ]
            }
        }

        # Which connection to use
        _alias = elastic_init(_args.env)

        sentence_search = OpinionSentence.search(using=_alias). \
            query("match", opinion_id=cited_opinion_id). \
            query("intervals", raw_text=sentence_query). \
            params(filter_path=["hits.hits._id", "hits.hits._source.sentence_id"])

        results = sentence_search.execute()
        sentence_elastic_id = results[0].meta["id"]
        sentence_id = results[0].sentence_id

        return {
            "sentence_elastic_id": sentence_elastic_id,
            "sentence_id": sentence_id
        }

    @queue_worker
    def _search_sentence_from_elastic(list_citation_graph: List[OpinionCitationGraph]) -> int:
        """
        Gets a list of OpinionCitationGraph and for each verbatim identify which sentence it was in the
        cited opinion.
        Uses multithreading to launch the queries in parallel. This improves the throughput of the program.
        :param list_citation_graph:
        :return:
        """
        threads = []
        updates_opinion_sentence = []
        updates_opinion_citation_graph = []

        def _create_thread(op_citation_graph: OpinionCitationGraph):
            def _thread_target():
                try:
                    data = _search_sentence(op_citation_graph.cited_opinion_id, op_citation_graph.verbatim)
                    updates_opinion_citation_graph.append(
                        {"citation_elastic_id": op_citation_graph.meta["id"],
                         "sentence_id": data["sentence_id"]}
                    )
                    updates_opinion_sentence.append(
                        {"sentence_elastic_id": data["sentence_elastic_id"],
                         "update_count": 1}
                    )
                except Exception:
                    return
            return _thread_target

        for ocg in list_citation_graph:
            threads.append(Thread(target=_create_thread(ocg)))

        # Launch all threads
        for t in threads:
            t.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Store results in tmp files
        write_jsonl(updates_opinion_sentence, tmpfile_sentence, append=True)
        write_jsonl(updates_opinion_citation_graph, tmpfile_citation, append=True)

        return len(list_citation_graph)

    @queue_worker
    def _search_sentence_from_parquet(x: pa.RecordBatch) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        batch: pd.DataFrame = x.to_pandas()
        updates_opinion_sentence = []

        for _, d in batch.iterrows():
            d: pd.Series
            try:
                data = _search_sentence(d['cited_opinion_id'], d['verbatim'])
            except Exception:
                continue

            updates_opinion_sentence.append({"sentence_elastic_id": data["sentence_elastic_id"], "update_count": 1})

        # Store results in tmp files
        write_jsonl(updates_opinion_sentence, tmpfile_sentence, append=True)

        return x.num_rows

    @queue_worker
    def _update_sentences(x: List[Tuple[str, int]]) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        _alias = elastic_init(_args.env)

        for elastic_id, count_citations in x:
            s = OpinionSentence()
            s.meta.id = elastic_id
            s.update(using=_alias, highlight=True, count_citations=count_citations)
        return len(x)

    @queue_worker
    def _update_citationgraph(x: List[Dict[str, int]]) -> int:
        # Initialize the connection to Elasticsearch
        # Making use of elasticsearch_dsl persistence features
        _alias = elastic_init(_args.env)

        for update in x:
            s = OpinionCitationGraph()
            s.meta.id = update["citation_elastic_id"]
            s.update(using=_alias, sentence_id=update["sentence_id"])
        return len(x)

    def _consolidate_updates():
        # Gather all the updates
        list_updates_opinion_sentence = read_jsonl(tmpfile_sentence)
        list_updates_opinion_citation_graph = read_jsonl(tmpfile_citation)

        # Consolidate the sentence updates (citation counter) into a single dictionary
        updates_opinion_sentence = defaultdict(lambda: 0)
        for u in list_updates_opinion_sentence:
            updates_opinion_sentence[u["sentence_elastic_id"]] += u["update_count"]

        logging.info(f"OpinionSentences collected {len(updates_opinion_sentence)} updates.")
        logging.info(f"OpinionCitationGraph: collected {len(list_updates_opinion_citation_graph)} updates.")

        return {
            "opinion_sentence": updates_opinion_sentence,
            "opinion_citation_graph": list_updates_opinion_citation_graph
        }

    # Now we are done with defining all inner functions.
    # Here starts the code for the main process of search_verbatim_sentence()

    # Mutually exclusive options
    assert (_args.from_parquet is not None) == (not _args.from_elastic)

    if not _args.only_update:
        if _args.from_parquet is not None:
            logging.info(f'Processing the dataset in {_args.from_parquet}')
            dataset: Any = ds.dataset(_args.from_parquet)
            dataset: ds.FileSystemDataset
            iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=128)
            multiprocess(worker_fn=_search_sentence_from_parquet, input_iterator_fn=iterator, total=nb_rows,
                         nb_workers=_args.num_workers, description='Search Verbatims Sentences')
        else:
            logging.info(f'Processing the collection OpinionCitationGraph')
            cache_file = "citationgraph.cache.pickle"
            if os.path.isfile(cache_file):
                with open(cache_file, "rb") as src:
                    logging.info(f"Reading cache file {cache_file}")
                    verbatims = pickle.load(src)
            else:
                alias = elastic_init(_args.env)
                verbatims_search = OpinionCitationGraph.search(using=alias).filter("range", score={"gt": -1})
                nb_verbatims = verbatims_search.count()
                verbatims = list(
                    tqdm.tqdm(verbatims_search.scan(), total=nb_verbatims, desc="Collecting from ElasticSearch"))
                with open(cache_file, "wb") as out:
                    logging.info(f"Caching result in {cache_file}...")
                    pickle.dump(verbatims, out)

            iterator, nb_verbatims = batch_iterator(verbatims, batch_size=_args.batch_size)
            multiprocess(worker_fn=_search_sentence_from_elastic, input_iterator_fn=iterator, total=nb_verbatims,
                         nb_workers=_args.num_workers, description='Search Verbatim Sentences')

    updates = _consolidate_updates()

    logging.info("Update OpinionCitationGraph collection")
    iterator, nb_rows = batch_iterator(updates["opinion_citation_graph"], batch_size=128)
    multiprocess(worker_fn=_update_citationgraph, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description="Update OpinionCitationGraph.")

    if not _args.update_only_verbatim:
        logging.info(f"Updating OpinionSentence collection.")
        iterator, nb_rows = batch_iterator(list(updates["opinion_sentence"].items()), batch_size=128)
        multiprocess(worker_fn=_update_sentences, input_iterator_fn=iterator, total=nb_rows,
                     nb_workers=_args.num_workers, description="Update OpinionSentence")


def export_sentence_to_parquet():
    """
    Export the complete sentence-level annotated dataset.

    :return:
    """
    collection_to_parquet(OpinionSentence, envfile=_args.env, destination=_args.dest, batch_size=_args.batch_size)


def parse_args():
    argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    parser_index = subparsers.add_parser(name='index-opinions', description='Add all opinions to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_opinions)

    parser_index = subparsers.add_parser(name='index-sentences', description='Add each sentence of each opinion '
                                                                             'to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_sentences)

    parser_index = subparsers.add_parser(name='index-graph', description='Add the citation graph edges to the index')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_citation_graph)

    parser_search = subparsers.add_parser(name='search-verbatims', description='Search the verbatim quotes.')
    parser_search.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_search.add_argument('--dest', type=str, help='Destination Path for the PARQUET dataset')
    parser_search.add_argument('--max-words-extract', type=int, default=100, help='Number of words to consider around '
                                                                                  'a citation')
    parser_search.add_argument('--min-verbatim', type=int, default=5, help='Minimum number of words between quotes to'
                                                                           'consider it as a verbatim quote')
    parser_search.add_argument('--save', default=False, action='store_true', help='Save the YES/NO in Destination')
    parser_search.add_argument('--env', type=str, help='Path to the env file')
    parser_search.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_search.set_defaults(func=search_verbatims)

    parser_search = subparsers.add_parser(name='search-sentences',
                                          description='Search the exact sentence the verbatim quotes originate from.'
                                                      'Uses the OpinionCitationGraph index or a PARQUET data')
    parser_search.add_argument('--from-parquet', type=str, help='Use a PARQUET dataset')
    parser_search.add_argument('--from-elastic', default=False, action='store_true',
                               help='Use the OpinionCitationGraph collection, provide ENV file')
    parser_search.add_argument('--batch-size', type=int, default=1000,
                               help='Batch size for processing Elastic collection')
    parser_search.add_argument('--update-only-verbatim', default=False, action='store_true',
                               help='Update only the OpinionCitationGraph')
    parser_search.add_argument('--only-update', default=False, action='store_true',
                               help='No search, just update (will use the cached pickled data')
    parser_search.add_argument('--tmp-sentence', type=str, help="Folder where updates are temporarily stored")
    parser_search.add_argument('--tmp-citation', type=str, help="Folder where updates are temporarily stored")
    parser_search.add_argument('--env', type=str, help='Path to the env file')
    parser_search.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_search.set_defaults(func=search_verbatim_sentence)

    parser_export = subparsers.add_parser(name='export-sentence', description='Export the complete ElasticSearch '
                                                                              'collection OpinionSentence to a PARQUET'
                                                                              'dataset')
    parser_export.add_argument('--env', type=str, help='Path to the env file')
    parser_export.add_argument('--dest', type=str, help='Destination Folder for the PARQUET dataset')
    parser_export.add_argument('--batch-size', type=int, default=1000000, help='Number of records per PARQUET file')
    parser_export.set_defaults(func=export_sentence_to_parquet)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()
    logging.info('Done.')
    sys.exit()


if __name__ == '__main__':
    main()
