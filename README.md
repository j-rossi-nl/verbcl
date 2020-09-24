# Court Listener
Working with CourtListener dataset.

# Parquet Dataset
Throughout the program, datasets will be stored as PARQUET dataset.

See [pyarrow](https://arrow.apache.org/docs/index.html)

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
`python prepare_data.py download --to <FOLDER> --untar`

## From `*.tar.gz` files to PARQUET (parallel)
`python prepare_data.py opinion --path <FOLDER WITH TAR.GZ FILES> --tags id html_with_citations --dest <FOLDER FOR PARQUET DATASET>`
Takes the `tag.gz` files in the PATH folder (these files come from untaring the dataset after downloading it), and transform it
as a PARQUET dataset (ie a collection of PARQUET files) located in the folder DEST.
* `--tags` accepts a list of tags. Each opinion is originally a JSON file, these tags are the keys of the JSON file. The recommended
setting is to use only `id` and `html_with_citations`, but others are possible. See an example JSON file to figure out if
other keys are relevant.

## Create citation map (parallel)
`python prepare_data.py citation-map --path <FOLDER WITH PARQUET DATASET> --dest <FOLDER>`
Create a PARQUET dataset in folder DEST with all the citation relations between opinions. Each datapoint is a couple `(citing_opinion_id, cited_opinion_id)`

## Extract a random sample
`python prepare_data.py sample --path <FOLDER> --dest <FOLDER> --citation-map <CSV Citation Map> -- num-samples <NUMBER OF RANDOM SAMPLES>
--add-cited --add-citing`
This will create a new PARQUET file in the DEST folder, named `sample_code.parq`. This file contains opinions taken randomly from the
dataset in the PATH folder. 

There are 2 additional options:
* `--add-cited` will create a single PARQUET file in the DEST folder, named `sample_cited.parq`, that contains all the opinions that are
cited by opinions in the random sample.
* `--add-citing` will do the same for opinions that cite any opinion in the random sample

## Summarize (parallel)
`python prepare_data.py summary --path <FOLDER> --dest <FOLDER> --method <METHOD>`
This will create a new PARQUET dataset (ie a collection of PARQUET files) in the folder DEST. Each opinion full text is summarized, 
following the method indicated by the option `--method`.

* `--method` the list of methods can be queried in the usage. By default, `'textrank'` is assumed

## Create data for annotation (parallel)
`python prepare_data.py doccano --path <FOLDER> --dest <FOLDER> --max-words-extract <NUMBER>`
This will take all opinions in the PARQUET dataset in folder PATH, and transform each citation into a JSON compatible for input to 
[doccano](https://github.com/doccano/doccano). There will be multiple JSON files. For each document, the place where the citation
is made in the original text is labeled as `'CITATION'` (or the constant `opinion.DOCCANO_CITATION_TAG`)
* `--max-words-extract` decides how many words before and after the citation should be extracted for annotation. If no value is given, then the whole text is provided. 
 
 