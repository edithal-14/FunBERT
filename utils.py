"""
Utilities for training BERT on DLND novelty detection corpus
"""

import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import tensor
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainerCallback
)

LOG = logging.getLogger()
TRANSFORMERS_CACHE_DIR = '/home1/tirthankar/Vignesh/transformers_models'

def init_logger(logger, filename, console_logs=True):
    """
    Initialize logger object to print to console (optional)
    and write to a file
    """
    log_format = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if console_logs:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)

class LogCallback(TrainerCallback):
    """
    Trainer Callback to write logs to a file
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            LOG.debug(logs)

def create_model():
    """
    Create Distilled BERT model for Sequence Classification
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        cache_dir=TRANSFORMERS_CACHE_DIR,
        local_files_only=True
    )
    # Freeze the BERT Encoder since we only want to train the head classifier
    for param in model.base_model.parameters():
        param.requires_grad = False
    model.train()
    return model

def compute_metrics(pred):
    """
    Compute metrics from the prediction output of the transformer
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }

class DlndBertDataset(Dataset):
    """
    Pytorch Dataset class which presents DLND data for transformers to consume
    """

    def __init__(self, data):
        self.data = data
        self.labels = self.data[-1]
        self.encodings = self.tokenize()

    def __getitem__(self, idx):
        item = {k: tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

    def tokenize(self):
        """
        Tokenize DLND data
        """
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            'distilbert-base-uncased',
            cache_dir=TRANSFORMERS_CACHE_DIR,
            local_files_only=True
        )
        return tokenizer(
            [
                ('. '.join(src_docs), '. '.join(tgt_docs))
                for src_docs, tgt_docs in zip(self.data[-3], self.data[-2])
            ],
            padding=True,
            truncation=True
        )
