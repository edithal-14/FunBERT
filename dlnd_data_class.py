"""
Class file for DLND Data
"""

import logging
import os
import random
from sklearn.model_selection import train_test_split
from utils import DlndBertDataset
import xml.etree.ElementTree as ET

LOG = logging.getLogger()

class DlndData():
    """
    Class to load and process DLND dataset
    """

    def __init__(
        self,
        random_seed=1234,
        train_ratio=0.8,
        dset_path='/home1/tirthankar/Vignesh/TAP-DLND-1.0_LREC2018_modified'
    ):
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.dset_path = dset_path
        src_docs, tgt_docs = self.list_docs()
        tgt_contents = DlndData.read_contents(tgt_docs)
        src_contents = DlndData.read_contents(src_docs)
        golds = DlndData.get_golds(tgt_docs)
        self.data = [src_docs, tgt_docs, src_contents, tgt_contents, golds]
        self.perform_oversample_minority()

    def list_docs(self):
        """
        List source and target documents
        """
        source_docs = list()
        tgt_docs = list()
        for genre in os.listdir(self.dset_path):
            genre_dir = os.path.join(self.dset_path, genre)
            if os.path.isdir(genre_dir):
                for topic in os.listdir(genre_dir):
                    topic_dir = os.path.join(genre_dir, topic)
                    tgt_dir = os.path.join(topic_dir, 'target')
                    source_dir = os.path.join(topic_dir, 'source')
                    for doc in os.listdir(tgt_dir):
                        if doc.endswith('.txt'):
                            tgt_docs.append([os.path.join(tgt_dir, doc)])
                    rel_docs = list()
                    for doc in os.listdir(source_dir):
                        if doc.endswith('.txt'):
                            rel_docs.append(os.path.join(source_dir, doc))
                    source_docs.append(rel_docs)
        # Expand source_docs to the same size as tgt_docs
        j = 0
        new_src_docs = []
        for doc in tgt_docs:
            tgt_topic = doc[0].split('/')[-3]
            src_topic = source_docs[j][0].split('/')[-3]
            if not tgt_topic == src_topic:
                j += 1
            new_src_docs.append(source_docs[j])
        return new_src_docs, tgt_docs

    @staticmethod
    def get_golds(docs):
        """
        Get the list of redundant (including partial) documents
        and use it populate golds answers for each document

        class labels:
        0: non-novel class
        1: novel class
        """
        golds = list()
        for doc in docs:
            for tag in ET.parse(doc[0][:-4] + '.xml').findall('feature'):
                if 'DLA' in tag.attrib:
                    if tag.attrib['DLA'] == 'Novel':
                        golds.append(1)
                    else:
                        golds.append(0)
        return golds

    @staticmethod
    def read_contents(docs):
        """
        Read contents of documents in 'docs' array
        """
        contents = list()
        for doc_list in docs:
            doc_content = list()
            for doc in doc_list:
                doc_content.append(open(doc, 'rb').read().decode('utf-8', 'ignore'))
            contents.append(doc_content)
        return contents

    def perform_oversample_minority(self):
        """
        Random oversampling of minority class in self.data
        """
        random.seed(self.random_seed)
        answers = self.data[-1]
        if len(answers) >= 2 * sum(answers):
            minority_class = 1
        else:
            minority_class = 0
        n_minority = len([answer for answer in answers if answer == minority_class])
        n_samples_add = len(answers) - (2 * n_minority)
        minority_ids = [idx for idx, answer in enumerate(answers) if answer == minority_class]
        random_ids = random.choices(minority_ids, k=n_samples_add)
        for item_no, _ in enumerate(self.data):
            self.data[item_no] += [self.data[item_no][idx] for idx in random_ids]

    def split_data(self, data, split_ratio):
        """
        Split data into two parts as per the split ratio
        """
        splitting = train_test_split(
            *data,
            train_size=split_ratio,
            random_state=self.random_seed,
            shuffle=True,
            stratify=data[-1]
        )
        train_ids = [idx for idx in range(len(splitting)) if idx % 2 == 0]
        test_ids = [idx for idx in range(len(splitting)) if idx % 2 == 1]
        train_data = [splitting[idx] for idx in train_ids]
        test_data = [splitting[idx] for idx in test_ids]
        return train_data, test_data

    def return_datasets(self):
        """
        Return train, valid and test pytorch datasets
        """
        train_data, test_data = self.split_data(self.data, self.train_ratio)
        train_data, valid_data = self.split_data(train_data, 0.9)
        # Log data set size details
        LOG.debug('Train ratio: %f', self.train_ratio)
        LOG.debug('Total data size: %d', len(self.data[0]))
        LOG.debug('Train data size: %d', len(train_data[0]))
        LOG.debug('Valid data size: %d', len(valid_data[0]))
        LOG.debug('Test data size: %d', len(test_data[0]))
        return DlndBertDataset(train_data), DlndBertDataset(valid_data), DlndBertDataset(test_data)
