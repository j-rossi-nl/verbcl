import glob
import logging
import os
import pandas as pd
import torch
import tqdm

from abc import ABC
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import logging as transformers_logging

from utils import list_iterator, multiprocess, queue_worker, random_name


transformers_logging.set_verbosity_warning()


class HighlightSentenceClassifier:
    """
    Documentation tbd
    """
    model = "distilbert-base-uncased"

    def __init__(self, save_folder, load_checkpoint: bool = False):
        if load_checkpoint:
            logging.info(f"Load model from {save_folder}.")
            self.seq_clf = AutoModelForSequenceClassification.from_pretrained(save_folder)
        else:
            logging.info(f"Create untrained model.")
            self.seq_clf = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=1)

        self.save_folder = save_folder

    def train(self, data_folder):
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        train_dataset = AnnotatedTextDataset(data_folder)

        training_args = TrainingArguments(
            output_dir=self.save_folder,
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=256,
            max_steps=100000,
            save_steps=10000,
            save_total_limit=5,
            fp16=True,
            dataloader_num_workers=1
        )

        trainer = Trainer(
            model=self.seq_clf,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    def test(self, data_folder, save_folder):
        test_dataset = TestDataset(data_folder)

        training_args = TrainingArguments(
            output_dir=self.save_folder,
            do_predict=True,
            per_device_eval_batch_size=512,
            fp16=True,
            dataloader_num_workers=4
        )

        trainer = Trainer(
            model=self.seq_clf,
            args=training_args,
        )

        output = trainer.predict(test_dataset)
        predictions = expit(output.predictions).flatten()
        uuids = test_dataset.uuids

        with open(os.path.join(save_folder, "bert.run"), "w") as out:
            for uuid, pred in zip(uuids, predictions):
                opinion_id, sentence_id = uuid.split("-")
                out.write(f"{opinion_id}\tQ0\ts{sentence_id}\t0\t{pred}\tBERT\n")


class PandasToDataset:
    """
    From the original Pandas dataframe, build a tensor dataset.
    """
    column_text = 'raw_text'
    column_label = 'highlight'
    uuid_join_columns = ["opinion_id", "sentence_id"]

    def __init__(self, save_folder: str, batch_size: int, nb_workers: int):
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.nb_workers = nb_workers

    def __call__(self, df: pd.DataFrame):
        @queue_worker
        def _tokenize(x: pd.DataFrame) -> int:
            tokenizer = AutoTokenizer.from_pretrained(HighlightSentenceClassifier.model)
            data = []
            for _, sample in x.iterrows():
                encodings = tokenizer(text=sample[PandasToDataset.column_text],
                                      truncation=True,
                                      padding="max_length",
                                      max_length=128,
                                      return_tensors="pt")
                item = encodings.data
                item["labels"] = torch.tensor(sample[PandasToDataset.column_label], dtype=torch.float)
                item["uuid"] = '-'.join(map(str, [sample[c] for c in PandasToDataset.uuid_join_columns]))
                data.append(item)

            torch.save(data, os.path.join(self.save_folder, f"{random_name(16)}.pt"))
            return x.shape[0]

        batch_size = self.batch_size
        num_samples = df.shape[0]
        batches = [df[n: n+batch_size] for n in range(0, num_samples, batch_size)]

        logging.info("Prepare Iterator")
        iterator, nb_batches = list_iterator(batches)
        multiprocess(worker_fn=_tokenize, input_iterator_fn=iterator, total=num_samples, nb_workers=self.nb_workers,
                     description="Prepare BERT data")


class AnnotatedTextDataset(IterableDataset, ABC):
    """
    Pytorch dataset for our sentences.
    All data is loaded in memory.
    """
    def __init__(self, folder: str):
        self.files = glob.glob(os.path.join(folder, "*.pt"))
        self.uuids = []

    def __iter__(self):
        for f in self.files:
            data = torch.load(f)
            for d in data:
                # Obviously the PandasToDataset is faulty, builds sample tensor [1, seq_len],
                # where [seq_len] was expected
                # Fix here to avoid rebuilding my dataset and risking delays on paper
                self.uuids.append(d["uuid"])
                yield {k: torch.ravel(v) for k, v in d.items() if isinstance(v, torch.Tensor)}


class TestDataset(Dataset, ABC):
    def __init__(self, folder: str):
        super().__init__()
        logging.info(f"Reading test data from {folder}")
        self.data = sum((torch.load(f) for f in tqdm.tqdm(glob.glob(os.path.join(folder, "*.pt")))), [])
        self.uuids = [x["uuid"] for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        return {k: torch.ravel(v) for k, v in self.data[item].items() if isinstance(v, torch.Tensor)}

