# Court Listener
Working with CourtListener dataset.

# Python Conda Environment
`conda env create -f environment.yml` will create an environment named `courtlistener`.

# Parquet Dataset
Throughout the program, datasets will be stored as PARQUET dataset.

See [pyarrow](https://arrow.apache.org/docs/index.html)

# Python Path
The environment variable `PYTHONPATH` should be set to the root folder of the repo, so `utils` is seen as a module and can be imported as `import utils`

# Prepare Data
This module is in charge of transforming the provided dataset of opinions in a few ways:
* Download the dataset
* Transform from the `*.tar.gz` files to a PARQUET dataset of opinions
* Create a citation map (who cites who)
* Extract a random sample of opinions from a dataset
* Summarize a dataset of opinions
* Create the material for annotation from a dataset of opinions

## Parallel
The code uses multiprocessing to do parallel processing of the data. When a command is indicated as (parallel), there is an additional available option:
* `--num-workers`: indicates how many parallel workers can be launched. The default is 4.

## Download
`PYTHONPATH=. python scripts/prepare_data.py download --to <FOLDER> --untar`

## From `*.tar.gz` files to PARQUET (parallel)
`PYTHONPATH=. python scripts/prepare_data.py opinion --path <FOLDER WITH TAR.GZ FILES> --tags id html_with_citations --dest <FOLDER FOR PARQUET DATASET>`
Takes the `tag.gz` files in the PATH folder (these files come from untaring the dataset after downloading it), and transform it
as a PARQUET dataset (ie a collection of PARQUET files) located in the folder DEST.
* `--tags` accepts a list of tags. Each opinion is originally a JSON file, these tags are the keys of the JSON file. The recommended
setting is to use only `id` and `html_with_citations`, but others are possible. See an example JSON file to figure out if
other keys are relevant.

## Create citation map (parallel)
`PYTHONPATH=. python scripts/prepare_data.py citation-map --path <FOLDER WITH PARQUET DATASET> --dest <FOLDER>`
Create a PARQUET dataset in folder DEST with all the citation relations between opinions. Each datapoint is a couple `(citing_opinion_id, cited_opinion_id)`

## Extract a random sample
`PYTHONPATH=. python scripts/prepare_data.py sample --path <FOLDER> --dest <FOLDER> --citation-map <CSV Citation Map> -- num-samples <NUMBER OF RANDOM SAMPLES>
--add-cited --add-citing`
This will create a new PARQUET file in the DEST folder, named `sample_code.parq`. This file contains opinions taken randomly from the
dataset in the PATH folder. 

There are 2 additional options:
* `--add-cited` will create a single PARQUET file in the DEST folder, named `sample_cited.parq`, that contains all the opinions that are
cited by opinions in the random sample.
* `--add-citing` will do the same for opinions that cite any opinion in the random sample

## Summarize (parallel)
`PYTHONPATH=. python scripts/prepare_data.py summary --path <FOLDER> --dest <FOLDER> --method <METHOD>`
This will create a new PARQUET dataset (ie a collection of PARQUET files) in the folder DEST. Each opinion full text is summarized, 
following the method indicated by the option `--method`.

* `--method` the list of methods can be queried in the usage. By default, `'textrank'` is assumed

## Create data for annotation (parallel)
`PYTHONPATH=. python scripts/prepare_data.py doccano --path <FOLDER> --dest <FOLDER> --max-words-extract <NUMBER>`
This will take all opinions in the PARQUET dataset in folder PATH, and transform each citation into a JSON compatible for input to 
[doccano](https://github.com/doccano/doccano). There will be multiple JSON files. For each document, the place where the citation
is made in the original text is labeled as `'CITATION'` (or the constant `opinion.DOCCANO_CITATION_TAG`)
* `--max-words-extract` decides how many words before and after the citation should be extracted for annotation. If no value is given, then the whole text is provided. 
 
## Create data for ProphetNet
`PYTHONPATH=. python scripts/prepare_data.py prophetnet --path <FOLDER>> --dest <FOLDER>>`
This will create 2 files:
* `opinions.txt` contains the raw text of legal opinion. One opinion per line in the file.
* `opinions.idx` contains the opinion_id of the opinions, in the same order as in `opinions.txt`

These 2 files can be moved to the ProphetNet repository, in the `src/courtlistener/original_data` folder, so that ProphetNet can be used for summarization with the `courtlistener_*.sh` scripts.
This is based on this [fork](https://github.com/j-rossi-nl/ProphetNet)
 
# Read opinion
`PYTHONPATH=. python scripts/show.py --path <FOLDER>`
Starts an interactive shell. At each round, the user can enter:
* either an opinion ID
* or `'stop'` to exit
 
After the opinion has been found it is displayed in a browser.  
 
 
# Verbatim Quotes
A verbatim quote happens when an opinion cites another opinion by providing an exact quote from the cited opinion. 
Potential quotes are spans of texts in between double quotes nearby a citation. 
 
## Elasticsearch
We use ElasticSearch to index all the opinions and identify where a span of text is actually a verbatim.
See [Elasticsearch](https://www.elastic.co/). We use 7.9.2.
 
The connection to the Elasticsearch node is configured through a `.env` file containing the following keys:
* Connection to a ELASTIC CLOUD instance:
* `ELASTIC_CLOUD_ID`
* `ELASTIC_CLOUD_API_ID`
* `ELASTIC_CLOUD_API_KEY`
* Connection to an instance
* `ELASTIC_HOST`
* `ELASTIC_PORT`

The location of this file is provided with the `--env` argument.

The index is always `juju-01`.
 
## Populate the Elasticsearch index
  `PYTHONPATH=. python scripts/elastic.py index --path <FOLDER> --env <FILE>`
  
Add each opinion in the data given by the `--path` argument to the index `juju-01`.
  
  
## Score 
`python elastic.py search --path <FOLDER> --dest <FOLDER> --envc<FILE>`
  
For each opinion in the dataset given by `--path`, associate each potential verbatim citation to a score, based on a query with Elasticsearch. The results dataset of citing/cited/verbatim/score is saved as a PARQUET dataset in `--dest`.


## Annotations
A website has been developed to annotate the verbatims.
To run it:
* `cd annotate_verbatims`
* adjust the path to the folders with your PARQUET dataset in the file `verbatims/apps.py`, by changing the class attributes of `VerbatimsConfig`. CORE is a dataset of opinions, CITED is a dataset that contains all the opinions cited in CORE
* adjust the path to the annotation file
* `PYTHONPATH=/home/juju/PycharmProjects/courtlistener/ python manage.py runserver` will start the server
* Use a browser to reach [link](http://localhost:8000/verbatims)
* The website presents a quote and the cited opinion, up to you to find if this is actually a verbatim quote from the cited opinion
* Not all words might be used, there are ellipsis (...) and so on
* At the end of annotation, all annotations are available in a JSON file, as setup in the settings. The JSON file is saved after each annoation.

The website uses (badly) Django. 
 