import itertools
import logging
import networkx as nx
import os
import pandas as pd
import sys
import utils

from argparse import ArgumentParser
from multiprocessing import Pool
from tqdm import tqdm

# Configure
utils.config()


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    # noinspection PyUnresolvedReferences
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        tqdm(zip(
            [G] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [True] * num_chunks,
            [None] * num_chunks,
        ), total=num_chunks),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


def parse_args(argstxt=None):
    if argstxt is None:
        argstxt = sys.argv[1:]
    parser = ArgumentParser()

    parser.add_argument("--graph", type=str, help="Pickle file with the Graph object.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--dest", type=str, help="Destination folder for CSV files generated.")
    return parser.parse_args(argstxt)


def main():
    args = parse_args()

    logging.info(f"Loading {args.graph}")
    g = nx.read_gpickle(args.graph)

    logging.info("Starting Computation...")
    bet_cn = betweenness_centrality_parallel(g, processes=args.num_workers)

    logging.info("Finished.")

    save_file = os.path.join(args.dest, "betweenness_centrality.parq")
    logging.info(f"Saving to {save_file}")
    pd.DataFrame(bet_cn).to_parquet(save_file)
        
    logging.info("Done.")


if __name__ == "__main__":
    main()
