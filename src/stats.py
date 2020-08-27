import pandas as pd
import numpy as np
import os
import glob
import json

DATA_FOLDER = '/ivi/ilps/personal/jrossi/cl_work'
CSV_SPLIT_FOLDER = os.path.join(DATA_FOLDER, '30_csv_split')

d = pd.read_csv(os.path.join(DATA_FOLDER, 'backrefs.csv'))
cits = pd.read_csv(os.path.join(DATA_FOLDER, 'citations.csv'))

sample_courts_csv = glob.glob(os.path.join(CSV_SPLIT_FOLDER, '*_0000.csv'))  # return N csv files


cits_unique_citing = cits['citing_opinion_id'].unique()
cits_unique_cited = cits['cited_opinion_id'].unique()

stats = {
    'initial_dataset_nb_opinions': len(d['opinion_id'].unique()),
    'nb_opinions_with_html_citations': len(cits_unique_citing),
    'nb_vertices_citation_graph': len(pd.Series(np.hstack([cits_unique_citing, cits_unique_cited])).unique()),
    'nb_edges_citation_graph': len(cits),
}

with open('stats.json', 'w') as outfile:
    outfile.write(json.dumps(stats))

for k, v in stats.items():
    print('{:<40}: {:>15,}'.format(k, v))