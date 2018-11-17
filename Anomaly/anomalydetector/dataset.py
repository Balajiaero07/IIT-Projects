import gzip
import logging
import os
import pickle as pickle
import time

import numpy as np

from anomalydetector import TMP_DIR, EVENTLOG_DIR
from anomalydetector.processmining import EventLog


class Dataset:
    def __init__(self, dataset_name=None, seed=None):
        self.total_examples = 0
        self.max_len = 0
        self.features = None
        self.targets = None
        self.train_targets = None
        self.labels = None

        self._event_log = None
        self.trace_lens = None
        self.num_attributes = None
        self.attribute_dims = None

        self.dataset_name = None

        # logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # seed
        if seed is not None:
            np.random.seed(seed)

        if dataset_name is not None:
            self.load(dataset_name)

    def load(self, dataset_name, size=0, train=False):
        self.dataset_name = dataset_name

        pickle_file = os.path.join(TMP_DIR, dataset_name + '.pkl.gz')
        event_log_file = os.path.join(EVENTLOG_DIR, dataset_name + '.json.gz')

        # check for pickle file and load from it
        if os.path.isfile(pickle_file):
            s = time.time()

            with gzip.open(pickle_file, 'rb') as f:
                self.features, self.targets, self.labels, self.trace_lens, self.attribute_dims = pickle.load(f)

            self.logger.info('Dataset "{}" loaded from cache in {:.4f}s'.format(dataset_name, time.time() - s))

        # else generate from event log
        elif os.path.isfile(event_log_file):
            s = time.time()

            # load data from event log
            self._event_log = EventLog.from_json(event_log_file)
            self.features, self.targets, self.labels, self.trace_lens, self.attribute_dims = self.from_event_log(
                self._event_log)

            # save to disk
            with gzip.open(pickle_file, 'wb') as f:
                pickle.dump((self.features, self.targets, self.labels, self.trace_lens, self.attribute_dims), f)

                self.logger.info(
                    'Dataset "{}" loaded and cached from event log in {:.4f}s'.format(dataset_name, time.time() - s))

        # when training targets need to be different
        if train:
            self.targets = [np.ones((self.features.shape[0], int(self.attribute_dims[i]))) for i in
                            range(len(self.attribute_dims))]
            oh = [np.eye(int(self.attribute_dims[j])) for j in range(len(self.attribute_dims))]
            for i, trace_len in enumerate(self.trace_lens):
                offset = np.random.choice(range(1, trace_len))

                # append the next onehot vector to be the label
                for j in range(self.features.shape[2]):
                    self.targets[j][i] = oh[j][int(self.features[i, offset, j] - 1)]

                # set the rest of features to be zeros (padding)
                self.features[i, offset:] = 0

        self.total_examples, self.max_len, self.num_attributes = self.features.shape
        if self.attribute_dims is None:
            self.attribute_dims = self.features.max(axis=0).max(axis=0)

        # return correct size as given by the parameter
        if 0 < size < self.total_examples:
            random_choice = np.random.choice(np.arange(self.features.shape[0]), size, replace=False)
            self.features = self.features[random_choice]
            if isinstance(self.targets, list):
                self.targets = [t[random_choice] for t in self.targets]
            else:
                self.targets = self.targets[random_choice]
            self.labels = self.labels[random_choice]
            self.trace_lens = self.trace_lens[random_choice]

        return self.features, self.targets

    @property
    def event_log(self):
        if self._event_log is None and self.dataset_name is not None:
            event_log_file = os.path.join(EVENTLOG_DIR, self.dataset_name + '.json.gz')
            self._event_log = EventLog.from_json(event_log_file)
            return self._event_log
        else:
            return self._event_log

    @staticmethod
    def from_event_log(event_log):
        features, trace_lens = event_log.to_feature_columns()
        labels = np.array([trace.attr['label'] for trace in event_log])
        targets = event_log.get_labels()
        attribute_dims = None
        if 'attr_dims' in event_log.attr:
            d = event_log.attr['attr_dims']
            attribute_dims = np.array([d[k] for k in sorted(d.keys())])
        return features, targets, labels, trace_lens, attribute_dims


class OneHotDataset(Dataset):
    def load(self, dataset_name, size=0, train=False):
        super().load(dataset_name=dataset_name, size=size, train=train)

        offsets = np.concatenate(([0], np.cumsum(self.attribute_dims)[:-1]))
        features = np.zeros([*self.features.shape[:-1], np.sum(self.attribute_dims, dtype=int)])
        for i, j, k in np.ndindex(*self.features.shape):
            if self.features[i, j, k] != 0:
                off = offsets[k]
                feat = self.features[i, j, k] - 1
                features[i, j, int(off + feat)] = 1

        self.features = features
        return self.features, self.targets


class FlatOneHotDataset(OneHotDataset):
    def load(self, dataset_name, size=0, train=False):
        super().load(dataset_name=dataset_name, size=size, train=False)
        self.features = self.features.reshape((self.features.shape[0], np.product(self.features.shape[1:])))
        return self.features
        model = os.path.basename(model)
        if '_' in model:
            dataset_name = model[:model.find('_')]
        else:
            dataset_name = model

        dataset = FlatOneHotDataset()
        x, y, labels = dataset.load(dataset_name)

        # get labels
        trace_labels = [any(l) for l in y]
        transition_labels = np.concatenate(y)

        # trace label probabilities
        self._labels, counts = np.unique(trace_labels, return_counts=True)
        self._trace_probabilities = counts / np.sum(counts)

        # transition label probabilities
        _, counts = np.unique(transition_labels, return_counts=True)
        self._transition_probabilities = counts / np.sum(counts)

#*************************

#def main():
#    x =OneHotDataset().load( dataset_name='huge-0.1-1')
#    print(x)

#if __name__ == '__main__':
#    main()