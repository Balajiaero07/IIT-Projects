import _pickle as pickle
import gzip
import os
import textwrap

import numpy as np
from sklearn.metrics import f1_score

from anomalydetector import EVAL_DIR, PLOT_OUT_DIR
from anomalydetector.anomalydetection import DAEAnomalyDetector
from anomalydetector.anomalydetection import HMMAnomalyDetector
from anomalydetector.anomalydetection import LSTMAnomalyDetector
from anomalydetector.anomalydetection import RNNGRUAnomalyDetector
from anomalydetector.anomalydetection import OneClassSVMAnomalyDetector
from anomalydetector.anomalydetection import SlidingWindowAnomalyDetector

# ADs and their respective datasets
AD = {
    LSTMAnomalyDetector(embedding=False).abbreviation: LSTMAnomalyDetector(embedding=False),
    LSTMAnomalyDetector(embedding=True).abbreviation: LSTMAnomalyDetector(embedding=True),
    DAEAnomalyDetector().abbreviation: DAEAnomalyDetector(),
    RNNGRUAnomalyDetector().abbreviation: RNNGRUAnomalyDetector(),
    HMMAnomalyDetector().abbreviation: HMMAnomalyDetector(),
    SlidingWindowAnomalyDetector().abbreviation: SlidingWindowAnomalyDetector(),
    OneClassSVMAnomalyDetector().abbreviation: OneClassSVMAnomalyDetector()
}


class Evaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.eventlog_name, self.ad_abbr = model_name.split('_')[:-1]
        self.test_eventlog_name = self.eventlog_name + '-test'

        self.ad = AD[self.ad_abbr]
        self.dataset = self.ad.dataset.__class__()
        self.anomaly_scores, self.train_anomaly_scores = self.get_anomaly_scores(model_name)
        self.dataset.load(self.test_eventlog_name, size=self.anomaly_scores.shape[0])

    def get_anomaly_scores(self, model_name):
        if self.cached_version_available(model_name):
            test_anomaly_scores, train_anomaly_scores = self.load_from_cache(model_name)
            return test_anomaly_scores, train_anomaly_scores
        else:
            self.ad.load(model_name)
            train_anomaly_scores = self.ad.predict_proba(self.eventlog_name)
            test_anomaly_scores = self.ad.predict_proba(self.test_eventlog_name)
            ret = (test_anomaly_scores, train_anomaly_scores)
            self.cache_file(self.get_cache_path(model_name), ret)
            return ret

    @staticmethod
    def get_cache_file_name(model_name):
        # remove extension
        model_name = os.path.splitext(model_name)[0]
        return '{}_evaluation.pkl.gz'.format(model_name)

    @staticmethod
    def get_cache_path(model_name):
        return os.path.join(EVAL_DIR, Evaluator.get_cache_file_name(model_name))

    @staticmethod
    def cache_file(path, anomaly_scores):
        with gzip.open(path, 'wb') as f:
            pickle.dump(anomaly_scores, f)

    @staticmethod
    def load_from_cache(model_name):
        return pickle.load(gzip.open(Evaluator.get_cache_path(model_name), 'rb'))

    @staticmethod
    def cached_version_available(model_name):
        return os.path.isfile(Evaluator.get_cache_path(model_name))

    @staticmethod
    def _flatten_axis(a, axis=2):
        if axis < 2:
            a = np.any(a == -1, axis=2) * -2 + 1
        if axis == 0:
            a = np.any(a == -1, axis=1) * -2 + 1
        return a

    def get_threshold(self, strategy='average'):
        threshold = self.train_anomaly_scores

        # threshold per event position
        if strategy in ['average', 'position']:
            threshold = threshold.mean(axis=0)

        # average threshold per attribute
        if strategy == 'average':
            threshold = threshold.mean(axis=0)

        return threshold

    @staticmethod
    def predict(anomaly_scores, thresholds, t):
        return (anomaly_scores <= (t * thresholds)) * -2 + 1

    def get_predictions(self, threshold='average', alpha=0.1, axis=2):
        y_true = self._flatten_axis(self.dataset.targets, axis=axis)
        y_pred = self._flatten_axis(
            Evaluator.predict(self.anomaly_scores, self.get_threshold(strategy=threshold), t=alpha), axis=axis)
        y_pred = np.ma.array(y_pred, mask=y_true.mask)
        return y_pred, y_true

    def evaluate(self, threshold='average', t=0.1, axis=2):
        y_pred, y_true = self.get_predictions(threshold, t, axis)
        a = f1_score(y_true.compressed(), y_pred.compressed(), average='macro')
        b = f1_score(y_true.compressed(), y_pred.compressed(), average=None, labels=[1, -1])
        return [a, b[0], b[1]]

    @staticmethod
    def _get_heatmap_objects(anomaly_scores, traces, labels, attribute_names):
        annot = np.empty_like(anomaly_scores, dtype=object)
        for i, trace in enumerate(traces):
            for j, event in enumerate(trace):
                for k, attr in enumerate(attribute_names):
                    if attr == 'name':
                        annot[i, j, k] = '{}\n({:.2f})'.format('\n'.join(textwrap.wrap(str(event.name), 12)),
                                                               anomaly_scores[i, j, k])
                    else:
                        annot[i, j, k] = '{}\n({:.2f})'.format('\n'.join(textwrap.wrap(str(event.attr[attr]), 12)),
                                                               anomaly_scores[i, j, k])
            annot[i, len(trace)] = ['EOS\n({:.2f})'.format(attr) for attr in anomaly_scores[i, len(trace)]]

        scores = anomaly_scores.reshape((anomaly_scores.shape[0], np.product(anomaly_scores.shape[1:])))
        annot = annot.reshape((annot.shape[0], np.product(annot.shape[1:])))

        # prepare labels
        def get_label(label):
            if label == 'normal':
                return 'normal'
            else:
                l = ['{}: {}'.format(key, value) for key, value in label['attr'].items() if
                     not key.startswith('_') and not isinstance(value, dict)]
                return '{}\n{}'.format(label['anomaly'], '\n'.join(l))

        labels = [get_label(l) for l in labels]

        return scores, annot, labels

    def plot_heatmap(self, file_name, sample_size=10, indices=None):
        import seaborn as sns
        sns.set_style('white')
        sns.set_context('notebook')
        sns.plt.switch_backend('Agg')

        if indices is None:
            indices = np.random.choice(self.dataset.total_examples, sample_size)

        scores = self.anomaly_scores[indices]
        trace_lens = self.dataset.trace_lens[indices] - 1
        labels = self.dataset.labels[indices]
        traces = np.array(self.dataset.event_log.traces)[indices]
        attribute_names = self.dataset.event_log.get_attribute_names()

        scores, annot, labels = self._get_heatmap_objects(scores, traces, labels, attribute_names)

        mask = np.zeros_like(scores)
        for i, trace_len in enumerate(trace_lens):
            mask[i, 2 * trace_len:] = 1

        width = int(2.5 * self.dataset.max_len)
        height = int(len(indices))
        fig, ax = sns.plt.subplots(figsize=(width, height))
        ax = sns.heatmap(scores, cmap='Blues_r', annot=annot, fmt='s',
                         square=False, cbar=True, mask=mask, linewidths=0.0, rasterized=True)

        # set x labels
        xticklabels = ['{} {}'.format(a, p) for p, a in
                       zip(np.repeat(np.arange(int(scores.shape[1] / 2)), len(attribute_names)),
                           np.tile(attribute_names, int(scores.shape[1] / 2)))]
        ax.set(xticklabels=xticklabels)
        ax.xaxis.set_ticks_position('top')

        # set y labels
        ax.set_yticklabels(labels=reversed(labels), rotation=0)

        sns.plt.tight_layout()
        fig.savefig(os.path.join(PLOT_OUT_DIR, file_name))
