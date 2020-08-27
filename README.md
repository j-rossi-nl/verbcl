# lawinsider
Working with LawInsider dataset

Pipeline :
Original dataset is available as *.tar.gz files : 
1 court = 1 tar.gz file (385 non-empty readable tar.gz) (extracted from CourtListener on September…) get the date

Each tar.gz file contains JSON files, 1 file = 1 opinion

Each opinion has many fields in JSON, including `html_with_citations` for opinions that cite another opinion.

Original data is in : `ivi/ilps/personal/jrossi/COURT_LISTENER`

The pipeline is as follows:  
1. Process all tar.gz file:
    1. `Court.tar.gz`  -> `court.csv`: 1 line per opinion, with a few fields, including `html_with_citations`  
    1. CSV files are in `cl_work/00_csv`    
    1. Command: 
    ```bash
    python prepare_data.py targztocsv --targzpath /ivi/ilps/personal/jrossi/COURT_LISTENER/opinions --dest /ivi/ilps/personal/jrossi/cl_work/00_csv --nbworkers=N
    ```
   
1. To prepare text for BERT pre-training:
    1. Process all CSV file to extract the full text from HTML
        1. `Court.csv -> court.txt` in folder `01/txt_raw`
        1. Command: 
        ```bash
        python prepare_data.py preparebert --csvpath /ivi/ilps/personal/jrossi/cl_work/00_csv --tag html_with_citations --dest /ivi/ilps/personal/jrossi/cl_work/01_txt_raw --nbworkers=N 
        ``` 
   
    1. Normalize text (lower, clean, strip, sentence tokenizer…)
    ```bash
    python cook_for_bert.py normalize --txtpath /ivi/ilps/personal/jrossi/cl_work/01_txt_raw --dest /ivi/ilps/personal/jrossi/cl_work/02_txt_normalized_unprocessed --nbworkers=N
    ```

    1. Split into smaller units (200k lines of text per file)
    ```bash
    python cook_for_bert.py split --txtpath /ivi/ilps/personal/jrossi/cl_work/02_txt_normalized_unprocessed --dest /ivi/ilps/personal/jrossi/cl_work/02_txt_normalized_split --size 200000 --nbworkers=N
    ```

    1. Build vocabulary file
        1. Sample lines from given files, the vocabulary is not based on the complete corpus. 
        1. Uses a BERT config file to read the total vocabulary size and the number of placeholders
        1. Uses SentencePiece module to train the tokenizer
        ```bash
        python cook_for_bert.py buildvocab --txtpath /ivi/ilps/personal/jrossi/cl_work/02_txt_normalized_split --spmprefix model --samplesize 20000000 --bertconfig bert_config.json --dest /ivi/ilps/personal/jrossi/cl_work/03_vocab --nbworkers=N
        ```

    1. Generate the TF record files for BERT pre-training
        1. The vocab and bert_config are in the same folder
        1. Needs a ‘bert’ folder with source code: 
        ```bash
       git clone https://github.com/google-research/bert
       python cook_for_bert.py generate --bertconfigpath /ivi/ilps/personal/jrossi/cl_work/03_vocab --bertsrcpath ~/repos/bert --txtpath /ivi/ilps/personal/jrossi/cl_work/02_txt_normalized_split -- pretrainpath /ivi/ilps/personal/jrossi/cl_work/04_tfrecords --nbworkers=N  
       ```

1. Extract gists from opinions (gist is the introduction to a citation of an opinion)
    1. `Court.csv `-> `court.json`
    1. The command can also collapse all JSON files into 1 big file and delete the court JSON files
    1. JSON comes from a panda dataframe: 
    ```python
    df.to_json(orient=’records’, lines=True)
    ```
    ```bash
    python prepare_data.py gist --method [nlp,last] --csvpath /ivi/ilps/personal/jrossi/cl_work/00_csv --gather /ivi/ilps/personal/jrossi/cl_work/10_catchphrases/gists.csv --dest /ivi/ilps/personal/jrossi/cl_work/10_catchphrases --json --nbworkers=N
    ```
    1. The big JSON file can be sorted in place
    ```bash
    python prepare_data.py sort --json=file
    ```
    
1. Create feature vectors for the gists, using BERT
    1. Tokenize all gists
        1. Select which opinions (by min / max number of citations)  or which specific opinion (by `opinion_id`)
        1. By `opinion_id`: 
        ```bash
        python get_data.py tokens --gist /ivi/ilps/personal/jrossi/cl_work/10_catchphrases/sorted_gists.json --cited_opid <id> --cache_tokens tokens --nb_workers=N
        ```
        1. By min-max 
        ```bash
        python get_data.py tokens --gist /ivi/ilps/personal/jrossi/cl_work/10_catchphrases/sorted_gists.json --min 20 --max 1000 --cache_tokens tokens_min_20_max_1000 --nb_workers=N
        ```
