"""
Fine tune distilled BERT model for sequence classification task using DLND data

Run in vignesh-bert environment
"""

from dlnd_data_class import DlndData
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(6)
import torch
import transformers
from utils import LogCallback, create_model, compute_metrics, init_logger


BATCH_SIZE = 32
LOG_FILE = 'dlnd_logs'
LOG = logging.getLogger()
init_logger(LOG, LOG_FILE)
transformers.logging.set_verbosity_debug()


if __name__ == '__main__':
    # Load data
    dlnd_train_dset, dlnd_valid_dset, dlnd_test_dset = DlndData().return_datasets()
    # Load model
    model = create_model()
    # Training
    training_args = transformers.TrainingArguments(
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        logging_dir='training_logs',
        logging_first_step=True,
        logging_steps=10,
        num_train_epochs=10,
        output_dir='training_results',
        per_device_eval_batch_size=BATCH_SIZE,
        per_device_train_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        metric_for_best_model='accuracy',
        disable_tqdm=True,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dlnd_train_dset,
        eval_dataset=dlnd_valid_dset,
        callbacks=[LogCallback],
    )
    trainer.train()
    trainer.evaluate()
