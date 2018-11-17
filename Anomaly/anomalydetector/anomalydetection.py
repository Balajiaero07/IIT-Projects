import _pickle as pickle
import datetime
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from anomalydetector import MODEL_OUT_DIR
from anomalydetector.dataset import Dataset
from anomalydetector.dataset import FlatOneHotDataset
from anomalydetector.dataset import OneHotDataset


class AnomalyDetector:
    """
    This is a boilerplate anomaly detector that only provides simple serialization and deserialization methods
    using pickle. Other classes can inherit the behavior. They will have to implement both the fit and the predict
    method.
    """

    def __init__(self, model=None, abbreviation=None):
        self.model = None

        self.abbreviation = abbreviation
        self.dataset = Dataset()

        if model is not None:
            self.load(model)

    def load(self, model):
        """
        Load a class instance from a pickle file. If no extension or absolute path are given the method assumes the
        file to be located inside the MODEL_OUT_DIR. It will also add the .pkl extension.

        :param model: path to the pickle file 
        :return: 
        """

        # set extension
        if not model.endswith('.pkl'):
            model += '.pkl'

        # set parameters
        if not os.path.isabs(model):
            model = os.path.join(MODEL_OUT_DIR, model)

        # load model
        self.model = pickle.load(open(model, 'rb'))

    def save(self, file_name):
        """
        Save the class instance using pickle.

        The filename will have the following structure: <file_name>_<self.abbreviation>_<current_datetime>.pkl

        :param file_name: custom file name 
        :return: 
        """
        if self.model:
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = '{}_{}_{}.pkl'.format(file_name, self.abbreviation, date)
            with open(os.path.join(MODEL_OUT_DIR, file_name), 'wb') as f:
                pickle.dump(self.model, f)
        else:
            raise Exception('No model has been trained yet.')

    def fit(self, eventlog_name):
        """
        This method must be implemented by the subclasses.

        :param eventlog_name: 
        :return: 
        """
        raise NotImplementedError()

    def predict_proba(self, eventlog_name):
        """
        This method must be implemented by the subclasses.

        :param eventlog_name: 
        :return: 
        """
        raise NotImplementedError()


class RandomAnomalyDetector(AnomalyDetector):
    def __init__(self, model=None):
        super().__init__(model=model, abbreviation='baseline')

        self._labels = None
        self._trace_probabilities = None
        self._transition_probabilities = None

    def fit(self, model):
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

    def predict_proba(self, traces):
        predictions = np.random.choice(self._labels, size=(traces.shape[0], traces.shape[1] - 1),
                                       p=self._trace_probabilities)
        return predictions


class SlidingWindowAnomalyDetector(AnomalyDetector):
    def __init__(self, model=None, k=2):
        super().__init__(model=model, abbreviation='sw')

        self.transitions = None
        self.threshold = None
        self.k = k

    def fit(self, traces):
        self.transitions = {}
        num_transitions = 0
        for trace in traces:
            for transition in zip(*[trace[i:] for i in range(self.k)]):
                t = '--'.join(transition)
                num_transitions += 1
                if t in self.transitions.keys():
                    self.transitions[t]['probability'] += 1
                else:
                    self.transitions[t] = {
                        'probability': 1,
                        'transition': transition
                    }

        for key in self.transitions.keys():
            self.transitions[key]['probability'] /= num_transitions

        self.threshold = np.mean([transition['probability'] for transition in self.transitions.values()])

    def predict_proba(self, traces):
        trace_lens = [len(trace) for trace in traces]
        num_windows = np.max(trace_lens) - (self.k - 1)

        trace_pred = np.empty((len(traces), num_windows))
        trace_pred[:] = np.infty
        for i, trace in enumerate(traces):
            for j, transition in enumerate(zip(*[trace[i:] for i in range(self.k)])):
                t = '--'.join(transition)
                if t in self.transitions:
                    trace_pred[i, j] = self.transitions[t]['probability']
                else:
                    trace_pred[i, j] = 0
        return trace_pred


class OneClassSVMAnomalyDetector(AnomalyDetector):
    def __init__(self, model=None):
        super().__init__(model=model, abbreviation='one-class-svm')

    def fit(self, traces):
        from sklearn.svm import OneClassSVM
        self.model = OneClassSVM(nu=0.8, kernel='poly')
        self.model.fit(traces)

    def predict(self, traces):
        # This only returns predictions for traces
        return self.model.predict(traces)


class HMMAnomalyDetector(AnomalyDetector):
    def __init__(self, n_components=4, model=None):
        super().__init__(model=model, abbreviation='hmm')

        self.n_components = n_components

    def fit(self, traces, trace_lens):
        from hmmlearn.hmm import GaussianHMM
        self.model = GaussianHMM(n_components=self.n_components, covariance_type="diag", n_iter=100)
        self.model.fit(traces, trace_lens)

    def predict(self, traces, trace_lens):
        x = np.split(traces, np.cumsum(trace_lens)[:-1])

        log_probs = []
        for seq in x:
            log_probs.append(self.model.decode(seq)[0])

        return np.array(log_probs)


class NNAnomalyDetector(AnomalyDetector):
    def __init__(self, model=None, abbreviation=None):
        super().__init__(model=model, abbreviation=abbreviation)

    def load(self, model):
        # set extension
        if not model.endswith('.h5'):
            model += '.h5'

        # set parameters
        if not os.path.isabs(model):
            model = os.path.join(MODEL_OUT_DIR, model)

        # load model
        from keras.models import load_model
        try:
            self.model = load_model(model)
        except Exception:
            print(model, 'failed to load')

    def save(self, dataset_name=None):
        if self.model:
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = '{}_{}_{}.h5'.format(dataset_name, self.abbreviation, date)
            self.model.save(os.path.join(MODEL_OUT_DIR, file_name))
            print('model saved')
        else:
            raise Exception('No net has been trained yet.')


class DAEAnomalyDetector(NNAnomalyDetector):
    def __init__(self, model=None):
        super().__init__(model=model, abbreviation='dae')

        self.dataset = FlatOneHotDataset()

    def fit(self, eventlog_name):
        import tensorflow as tf
        from tensorflow.contrib.keras.python.keras.engine import Input, Model
        from tensorflow.contrib.keras.python.keras.layers import Dense, GaussianNoise, Dropout

        # load data
        features = self.dataset.load(eventlog_name)

        # parameters
        input_size = features.shape[1]
        hidden_size = np.round(input_size * 4)

        # input layer
        input_layer = Input(shape=(input_size,), name='input')

        # hidden layer
        hid = Dense(hidden_size, activation=tf.nn.relu)(GaussianNoise(0.1)(input_layer))
        hid = Dense(hidden_size, activation=tf.nn.relu)(Dropout(0.5)(hid))
        hid = Dense(hidden_size, activation=tf.nn.relu)(Dropout(0.5)(hid))
        hid = Dense(hidden_size, activation=tf.nn.relu)(Dropout(0.5)(hid))
        hid = Dense(hidden_size, activation=tf.nn.relu)(Dropout(0.5)(hid))

        # output layer
        output_layer = Dense(input_size, activation='linear')(Dropout(0.5)(hid))

        # build model
        self.model = Model(inputs=input_layer, outputs=output_layer)

        # compile model
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
            loss=tf.losses.mean_squared_error
        )

        # train model
        self.model.fit(
            features,
            features,
            batch_size=100,
            epochs=100,
            validation_split=0.2,
        )

    def predict_proba(self, eventlog_name):
        """
        Calculate the anomaly score for each event attribute in each trace. 
        Anomaly score here is the mean squared error.

        :param traces: traces to predict 
        :return: 
            anomaly_scores: anomaly scores for each attribute; 
                            shape is (#traces, max_trace_length - 1, #attributes)

        """

        features = self.dataset.load(eventlog_name)

        # get event length
        event_len = np.sum(self.dataset.attribute_dims - 1).astype(int)

        # init anomaly scores array
        anomaly_scores = np.zeros((features.shape[0], self.dataset.max_len - 1, len(self.dataset.attribute_dims)))

        # get predictions
        predictions = self.model.predict(features)
        errors = (predictions - features) ** 2

        # remove the BOS event
        errors = errors[:, event_len:]

        # split the errors according to the attribute dims
        split = np.cumsum(np.tile(self.dataset.attribute_dims - 1, self.dataset.max_len - 1), dtype=int)[:-1]
        errors = np.split(errors, split, axis=1)
        errors = np.array([np.mean(a, axis=1) for a in errors])

        for i in range(len(self.dataset.attribute_dims)):
            error = errors[i::len(self.dataset.attribute_dims)]
            anomaly_scores[:, :, i] = error.T

        # TODO: Normalize the anomaly_scores to lie between 0 and 1
        return -anomaly_scores


class LSTMAnomalyDetector(NNAnomalyDetector):
    def __init__(self, model=None, embedding=True):
        self.embedding = embedding
        self.distributions = None

        super().__init__(model, abbreviation='lstm-emb' if embedding else 'lstm')

        if not self.embedding:
            self.dataset = OneHotDataset()
        else:
            self.dataset = Dataset()

    def load(self, model):
        # TODO: perhaps there is a better way to do this. We are probably gonna kill off the non-embedding version anyway.
        super().load(model)
        if 'lstm-emb' not in model:
            self.embedding = False
            self.dataset = OneHotDataset()

    def fit(self, eventlog_name):
        import tensorflow as tf
        from tensorflow.contrib.keras.python.keras.engine import Input, Model
        from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, LSTM, Embedding, merge, Masking

        # load data
        features, targets = self.dataset.load(eventlog_name, train=True)

        # input layers
        inputs = []
        layers = []

        if self.embedding:
            with tf.device('/cpu:0'):
                # split attributes
                features = [features[:, :, i] for i in range(features.shape[2])]

                for i, t in enumerate(features):
                    voc_size = np.array(self.dataset.attribute_dims[i]) + 1  # we start at 1, hence +1
                    emb_size = np.floor(voc_size / 2.0).astype(int)
                    i = Input(shape=(None, *t.shape[2:]))
                    x = Embedding(input_dim=voc_size, output_dim=emb_size, input_length=t.shape[1], mask_zero=True)(i)
                    inputs.append(i)
                    layers.append(x)

                # merge layers
                x = merge.concatenate(layers)

        else:
            # input layer
            i = Input(shape=(None, *features.shape[2:]))
            x = Masking(mask_value=0)(i)
            inputs.append(i)

        # LSTM layer
        x = LSTM(64, implementation=2)(x)
        

        # shared hidden layer
        x = Dense(512, activation=tf.nn.relu)(x)
        x = Dense(512, activation=tf.nn.relu)(Dropout(0.5)(x))

        # hidden layers per attribute
        outputs = []
        for i, l in enumerate(targets):
            o = Dense(256, activation=tf.nn.relu)(Dropout(0.5)(x))
            o = Dense(256, activation=tf.nn.relu)(Dropout(0.5)(o))
            o = Dense(l.shape[1], activation=tf.nn.softmax)(Dropout(0.5)(o))
            outputs.append(o)

        # build model
        self.model = Model(inputs=inputs, outputs=outputs)

        # compile model
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
            loss='categorical_crossentropy'
        )

        # train model
        self.model.fit(
            features,
            targets,
            batch_size=100,
            epochs=100,
            validation_split=0.2,
        )

    def predict_proba(self, eventlog_name):
        """
        Calculate the anomaly score and the probability distribution for each event in each trace.
        Anomaly score here is the probability of that event occurring given all events before.

        :param traces: traces to predict 
        :return: 
            anomaly_scores: anomaly scores for each attribute; 
                            shape is (#traces, max_trace_length - 1, #attributes)

            distributions: probability distributions for each event and attribute;
                           list of np.arrays with shape (#traces, max_trace_length - 1, #attribute_classes),
                           one np.array for each attribute, hence list len is #attributes
        """

        def _get_all_subsequences(sequence):
            """
            Calculate all subsequences for a given sequence after removing the padding (0s).

            :param sequence: 
            :return: 
            """

            num_subsequences = np.sum(np.any(sequence != 0, axis=1)) - 1  # remove padding and calculate num subseqs
            subsequences = np.zeros((num_subsequences, sequence.shape[0], sequence.shape[1]))  # init array
            next_events = sequence[1:num_subsequences + 1]  # get next event

            for i in np.arange(num_subsequences):
                length = num_subsequences - i
                subsequences[i, :length, :] = sequence[:length, :]

            return subsequences[::-1], next_events

        # load data
        features, _ = self.dataset.load(eventlog_name, train=False)

        # anomaly scores for attributes
        # shape is (#traces, max_len_trace - 1, #attributes)
        # we do not predict the BOS activity, hence the -1
        anomaly_scores = np.ones((features.shape[0], features.shape[1] - 1, len(self.dataset.attribute_dims)))

        # distributions for each attribute
        attr_dims = np.array([int(o.shape[1]) for o in self.model.output])
        self.distributions = [np.ones((features.shape[0], features.shape[1] - 1, attr_dim)) for attr_dim in attr_dims]

        sub_sequences = []
        next_events = []
        for i, trace in enumerate(features):
            s, n = _get_all_subsequences(trace)
            sub_sequences.append(s)
            next_events.append(n)

        sub_sequences = np.vstack(sub_sequences)
        next_events = np.vstack(next_events).astype(int)

        if self.embedding:
            sub_sequences = [sub_sequences[:, :, i] for i in range(sub_sequences.shape[2])]
            next_events = [next_events[:, i] - 1 for i in range(next_events.shape[1])]
        else:
            offset = np.concatenate([[0], np.cumsum(attr_dims)[:-1]])
            n = np.zeros((next_events.shape[0], attr_dims.shape[0]), dtype=int)
            for index, next_event in enumerate(next_events):
                n[index] = np.where(next_event == 1)[0] - offset
            next_events = [n[:, i] for i in range(n.shape[1])]

        cumsum = np.cumsum(self.dataset.trace_lens - 1)
        cumsum2 = np.concatenate(([0], cumsum[:-1]))
        offsets = np.dstack((cumsum2, cumsum))[0]
        dist = self.model.predict(sub_sequences)

        for i, _n in enumerate(next_events):
            scores = dist[i][range(dist[i].shape[0]), _n]
            for j, trace_len in enumerate(self.dataset.trace_lens - 1):
                start, end = offsets[j]
                anomaly_scores[j][:trace_len, i] = scores[start:end]
                self.distributions[i][j, :trace_len] = dist[i][start:end]

        return anomaly_scores


class RNNGRUAnomalyDetector(NNAnomalyDetector):
    def __init__(self, model=None, embedding=True):
        self.dataset = Dataset()
        super().__init__(model, abbreviation='RNNGRU')
        self.embedding = embedding

    def load(self, model):
        super().load(model)

    def fit(self, eventlog_name):

        import tensorflow as tf
        from tensorflow.contrib.keras.python.keras.engine import Input, Model
        from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, GRU, Embedding, merge, Masking

        features, targets = self.dataset.load(eventlog_name, train=True)
        inputs = []
        layers = []

        with tf.device('/cpu:0'):
            # split attributes
            features = [features[:, :, i] for i in range(features.shape[2])]

            for i, t in enumerate(features):
                voc_size = np.array(self.dataset.attribute_dims[i]) + 1  # we start at 1, hence +1
                emb_size = np.floor(voc_size / 2.0).astype(int)

                i = Input(shape=(None, *t.shape[2:]))
                x = Embedding(input_dim=voc_size, output_dim=emb_size, input_length=t.shape[1], mask_zero=True)(i)
                inputs.append(i)
                layers.append(x)

            # merge layers
            x = merge.concatenate(layers)

        x = GRU(64, implementation=2)(x)

        # shared hidden layer
        x = Dense(512, activation=tf.nn.relu)(x)
        x = Dense(512, activation=tf.nn.relu)(Dropout(0.5)(x))

        # hidden layers per attribute
        outputs = []
        for i, l in enumerate(targets):
            o = Dense(256, activation=tf.nn.relu)(Dropout(0.5)(x))
            o = Dense(256, activation=tf.nn.relu)(Dropout(0.5)(o))
            o = Dense(l.shape[1], activation=tf.nn.softmax)(Dropout(0.5)(o))
            outputs.append(o)

        self.model = Model(inputs=inputs, outputs=outputs)

        # compile model

        # old setting : optimizers from tensorflow

        # self.model.compile(
        # optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
        # loss='categorical_crossentropy'
        # )

        # new setting : optimizers from keras

        self.model.compile(
            optimizer='Adadelta',
            loss='categorical_crossentropy'
        )

        # train model
        self.model.fit(
            features,
            targets,
            batch_size=100,
            epochs=100,
            validation_split=0.2,
        )

    def predict_proba(self, eventlog_name):
        """
        Calculate the anomaly score and the probability distribution for each event in each trace.
        Anomaly score here is the probability of that event occurring given all events before.

        :param traces: traces to predict
        :return:
            anomaly_scores: anomaly scores for each attribute;
                    shape is (#traces, max_trace_length - 1, #attributes)

            distributions: probability distributions for each event and attribute;
                   list of np.arrays with shape (#traces, max_trace_length - 1, #attribute_classes),
                   one np.array for each attribute, hence list len is #attributes
        """

        def _get_all_subsequences(sequence):
            """
            Calculate all subsequences for a given sequence after removing the padding (0s).

            :param sequence:
            :return:
            """

            num_subsequences = np.sum(np.any(sequence != 0, axis=1)) - 1  # remove padding and calculate num subseqs
            subsequences = np.zeros((num_subsequences, sequence.shape[0], sequence.shape[1]))  # init array
            next_events = sequence[1:num_subsequences + 1]  # get next event

            for i in np.arange(num_subsequences):
                length = num_subsequences - i
                subsequences[i, :length, :] = sequence[:length, :]

            return subsequences[::-1], next_events

        # load data
        features, _ = self.dataset.load(eventlog_name, train=False)

        # anomaly scores for attributes
        # shape is (#traces, max_len_trace - 1, #attributes)
        # we do not predict the BOS activity, hence the -1
        anomaly_scores = np.ones((features.shape[0], features.shape[1] - 1, len(self.dataset.attribute_dims)))

        # distributions for each attribute
        attr_dims = np.array([int(o.shape[1]) for o in self.model.output])
        self.distributions = [np.ones((features.shape[0], features.shape[1] - 1, attr_dim)) for attr_dim in attr_dims]

        sub_sequences = []
        next_events = []

        for i, trace in enumerate(features):
            s, n = _get_all_subsequences(trace)
            sub_sequences.append(s)
            next_events.append(n)

        sub_sequences = np.vstack(sub_sequences)
        next_events = np.vstack(next_events).astype(int)

        if self.embedding:
            sub_sequences = [sub_sequences[:, :, i] for i in range(sub_sequences.shape[2])]
            next_events = [next_events[:, i] - 1 for i in range(next_events.shape[1])]
        else:
            offset = np.concatenate([[0], np.cumsum(attr_dims)[:-1]])
            n = np.zeros((next_events.shape[0], attr_dims.shape[0]), dtype=int)
            for index, next_event in enumerate(next_events):
                n[index] = np.where(next_event == 1)[0] - offset
                next_events = [n[:, i] for i in range(n.shape[1])]

        cumsum = np.cumsum(self.dataset.trace_lens - 1)
        cumsum2 = np.concatenate(([0], cumsum[:-1]))
        offsets = np.dstack((cumsum2, cumsum))[0]
        dist = self.model.predict(sub_sequences)

        for i, _n in enumerate(next_events):
            scores = dist[i][range(dist[i].shape[0]), _n]
            for j, trace_len in enumerate(self.dataset.trace_lens - 1):
                start, end = offsets[j]
                anomaly_scores[j][:trace_len, i] = scores[start:end]
                self.distributions[i][j, :trace_len] = dist[i][start:end]

        return anomaly_scores