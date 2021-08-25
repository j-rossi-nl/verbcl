# VerbCL

Here is the dataset from the paper: 
> VerbCL: A Dataset of Verbatim Quotes for Highlight Extraction in Case Law
>
> J. Rossi, S. Vakulenko, E. Kanoulas, 2021

## Build the Python Environment

We use `poetry` as dependency manager.

* Install `poetry` with `pip install poetry`
* Install dependencies with `poetry install`
* Install torch:
  * `CPU` version with `poetry run poe cpu`
  * `GPU CUDA 10.2` with `poetry run poe cuda102`
  * `GPU CUDA 11.1` with `poetry run poe cuda111`

This will create a new virtual environment. 
* Enter a shell where the environment is activated: `poetry shell`

## Download the VerbCL Data

The data is available: [Here](https://doi.org/10.21942/uva.14798878.v1).


## Restore Snapshot

* Python notebook for restore [here](notebooks/Load_Snapshot.ipynb) 
* DIY Instructions:
  * Uncompress the archive on your filesystem (e.g. `/data`)
  * Declare the data folder `/data/VerbCL` as the root of a Snapshot Repository [Instructions](https://www.elastic.co/guide/en/elasticsearch/reference/current/snapshots-register-repository.html)
  * Restore the snapshot `verbcl_v1.0` [Instructions](https://www.elastic.co/guide/en/elasticsearch/reference/current/snapshot-restore.html)

## Example notebooks

* Using the persistence API of `elasticsearch-dsl` in this [notebook](notebooks/Tutorial_Using_Data.ipynb)

# Reproduce the Paper

(tbd)
All these steps can be executed with our code:
1. Download court listener 
2. Prepare the dataset
3. Run baselines


# Citation
## Paper

Our paper is accepted at CIKM 2021, Resource Track.

* DOI: [10.1145/3459637.3482021](https://doi.org/10.1145/3459637.3482021)
* Pre-Print available on [ArXiv](https://arxiv.org/abs/2108.10120)


## Dataset
```
@misc{rossi-vakulenko-kanoulas-2021, 
  title={VerbCL Dataset}, 
  url={https://uvaauas.figshare.com/articles/dataset/VerbCL\_Dataset/14798878/1}, 
  DOI={10.21942/uva.14798878.v1}, 
  abstractNote={VerbCL is a dataset of US court opinions, where verbatim quotes have been mined.}, 
  publisher={University of Amsterdam / Amsterdam University of Applied Sciences}, 
  author={Rossi, J. and Vakulenko, S. and Kanoulas, E.}, 
  year={2021}, 
  month={Jun} 
} 
```
# Contact

For questions and inquiries, contact: [Julien Rossi](mailto:j.rossi@uva.nl)
