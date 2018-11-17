import _pickle as pickle
import gzip
import json
import os
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import untangle
from dateutil import parser

from anomalydetector import MODELS_DIR

# global variables
START_EVENT_NAME = 'BOS'
END_EVENT_NAME = 'EOS'


class Event:
    def __init__(self, name, start_time, end_time=None, **attr):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.attr = dict(attr)

    def __str__(self):
        return 'Event: name {}, timestamp: {}'.format(self.name, self.start_time)

    def __eq__(self, other):
        if isinstance(other, Event):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other

    def to_json(self):
        """Return the event object as a json compatible python dictionary."""
        return dict(name=self.name, start_time=self.start_time, end_time=self.end_time, attributes=self.attr)


class Trace:
    def __init__(self, trace_id=None, events=None, **attr):
        self.id = trace_id
        if events is None:
            self.events = []
        else:
            self.events = events
        self.attr = dict(attr)

    def __iter__(self):
        return iter(self.events)

    def __str__(self):
        return 'Trace {}: #events = {}'.format(
            self.id,
            self.get_num_events()
        )

    def __getitem__(self, index):
        return self.events[index]

    def __setitem__(self, index, value):
        self.events[index] = value

    def __len__(self):
        return len(self.events)

    def index(self, index):
        return self.events.index(index)

    def add_event(self, event):
        self.events.append(event)

    def get_num_events(self):
        return len(self.events)

    def to_name_list(self):
        return [str(event.name) for event in self.events]

    def to_json(self):
        """Return the trace object as a json compatible python dictionary."""
        return dict(id=self.id, events=[event.to_json() for event in self.events], attributes=self.attr)


class EventLog:
    def __init__(self, traces=None, **attr):
        if traces is None:
            self.traces = []
            self.activities = []
        else:
            self.traces = traces
            self.activities = list(
                set([event.name for trace in self.traces for event in trace]))
        self.variants = None
        self.variant_probabilities = None
        self.variant_counts = None
        self.attr = dict(attr)
        self.max_length = 0

    def __iter__(self):
        return iter(self.traces)

    def __str__(self):
        return 'Event Log: #traces: {}, #events: {}, #activities: {}'.format(
            len(self.traces),
            self.get_num_events(),
            len(self.activities)
        )

    def __getitem__(self, index, ):
        return self.traces[index]

    def __setitem__(self, index, value):
        self.traces[index] = value

    def get_attribute_names(self):
        return sorted([key for key in ['name'] + list(self.traces[0].events[0].attr.keys()) if not key.startswith('_')])

    def add_trace(self, trace):
        for event in trace:
            if event.name not in self.activities:
                self.activities.append(event.name)
        self.max_length = max(self.max_length, len(trace))
        self.traces.append(trace)

    def get_trace_lens(self):
        return np.array([trace.get_num_events() for trace in self.traces])

    def get_num_events(self):
        return np.sum(self.get_trace_lens())

    def get_variants(self):
        if not self.variants:
            self.variants, self.variant_counts = np.unique([trace.to_name_list() for trace in self.traces],
                                                           return_counts=True)
            self.variant_probabilities = self.variant_counts / float(len(self.traces))
        return self.variants, self.variant_counts, self.variant_probabilities

    def get_variants_with_atts(self):
        if not self.variants:
            self.variants, self.variant_counts = np.unique(
                [trace.as_list_with_atts() for trace in self.traces if trace.attr['label'] == 'normal'],
                return_counts=True)
            self.variant_probabilities = self.variant_counts / float(len(self.traces))
        return self.variants, self.variant_counts, self.variant_probabilities

    def get_labels(self):
        # +1 for EOS and +1 for the activity name
        labels = np.ones((len(self.traces), max(self.get_trace_lens()) + 1, len(self.get_attribute_names())))

        # set padding and mask
        mask = np.zeros_like(labels)
        for i, j in enumerate(self.get_trace_lens() + 1):
            labels[i, j:] = 0
            mask[i, j:] = 1

        for i, label in enumerate([trace.attr['label'] for trace in self.traces]):
            # set labels to true where the anomaly happens
            if isinstance(label, dict):
                anomaly_type = label['anomaly']
                if anomaly_type in ['DuplicateSequence', 'SkipSequence']:
                    labels[i, label['attr']['start']] = -1
                elif anomaly_type in ['IncorrectLongTermDependency', 'IncorrectAttribute']:
                    labels[i, label['attr']['index'], self.get_attribute_names().index(label['attr']['affected'])] = -1
                elif anomaly_type == 'SwitchEvents':
                    labels[i, label['attr']['first']] = -1
                    labels[i, label['attr']['first'] + 1] = -1
                    labels[i, label['attr']['second']] = -1
                    labels[i, label['attr']['second'] + 1] = -1
                elif anomaly_type == 'MissingHead':
                    labels[i, 0] = -1
                elif anomaly_type == 'MissingTail':
                    labels[i, -1] = -1

        return np.ma.array(labels, mask=mask, dtype=int)

    def to_json(self, file_path):
        """
        Save the event log to a JSON file.
        
        :param file_path: absolute path for the JSON file 
        :return: 
        """
        event_log = {"traces": [trace.to_json() for trace in self.traces], "attributes": self.attr}
        with gzip.open(file_path, 'wt') as outfile:
            json.dump(event_log, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    def to_csv(self, file_path):
        """
        Save the event log to a CSV file.
        
        :param file_path: absolute path for the CSV file
        :return: 
        """
        if not file_path.endswith('.csv'):
            '.'.join((file_path, 'csv'))
        df = self.to_dataframe()
        df.to_csv(file_path + '.csv', index=False)

    def to_dataframe(self):
        """
        Return pandas DataFrame containing the event log in matrix format.
        
        :return: pandas.DataFrame 
        """
        frames = []
        for trace_id, trace in enumerate(self.traces):
            if trace.id is not None:
                trace_id = trace.id
            for event in trace:
                frames.append({
                    'trace_id': trace_id,
                    'event': event.name,
                    'start_time': event.start_time,
                    'end_time': event.end_time,
                    **dict([i for i in event.attr.items() if not i[0].startswith('_')])
                })
        return pd.DataFrame(frames)

    def to_feature_columns(self):
        """
        Return current event log as feature columns.
        
        Attributes are integer encoded. Shape of feature columns is (#traces, max_len, #attributes).
         
        :return: feature_columns, trace_lens
        """
        feature_columns = {
            'name': [],
            'trace_lens': []
        }

        # TODO: This will break with numerical attributes
        bos = Event(start_time=None, end_time=None,
                    **dict((attr, START_EVENT_NAME) for attr in self.get_attribute_names()))
        eos = Event(start_time=None, end_time=None,
                    **dict((attr, END_EVENT_NAME) for attr in self.get_attribute_names()))

        for trace in self.traces:
            feature_columns['trace_lens'].append(len(trace) + 2)
            for event in [bos] + trace.events + [eos]:
                for attribute in self.get_attribute_names():
                    if attribute == 'name':
                        attr = event.name
                    else:
                        attr = event.attr[attribute]
                    if attribute not in feature_columns.keys():
                        feature_columns[attribute] = []
                    feature_columns[attribute].append(attr)

        label_encoders = []
        for key in self.get_attribute_names():
            from sklearn.preprocessing import LabelEncoder
            if 'attr_dims' in self.attr:
                dim = self.attr['attr_dims'][key]
                fit = [START_EVENT_NAME, END_EVENT_NAME]
                if key == 'name':
                    fit += self.activities
                else:
                    fit += [str(i) for i in range(dim - 2)]
            else:
                fit = feature_columns[key]
            enc = LabelEncoder()
            enc.fit(fit)
            feature_columns[key] = enc.transform(feature_columns[key]) + 1
            label_encoders.append(enc)

        # transform back into sequences
        trace_lens = np.array(feature_columns['trace_lens'])
        offsets = np.concatenate(([0], np.cumsum(trace_lens)[:-1]))
        features = np.zeros((len(trace_lens), np.max(trace_lens), len(self.get_attribute_names())))
        for i, (offset, trace_len) in enumerate(zip(offsets, trace_lens)):
            for k, key in enumerate(self.get_attribute_names()):
                x = feature_columns[key]
                features[i, :trace_len, k] = x[offset:offset + trace_len]

        return features, trace_lens

    @staticmethod
    def from_json(file_path):
        """
        Parse event log from JSON.

        JSON can be gzipped

        :param file_path: path to json file
        :return:
        """
        if file_path.endswith('gz'):
            import gzip
            open = gzip.open

        # read the file
        with open(file_path, 'rb') as f:
            log = json.loads(f.read().decode('utf-8'))

        event_log = EventLog(**log['attributes'])

        for trace in log['traces']:
            _trace = Trace(trace_id=trace['id'], **trace['attributes'])
            for e in trace['events']:
                event = Event(
                    name=e['name'], start_time=e['start_time'], end_time=e['end_time'], **e['attributes'])
                _trace.add_event(event)
            event_log.add_trace(_trace)

        return event_log

    @staticmethod
    def from_xes(file_path):
        """
        Load an event log from an XES file

        :param file_path: path to xes file
        :return: EventLog object
        """

        # TODO: Add attributes to the event log from the XES

        # TODO: Add check for zip as well
        # check for tar.gz
        if file_path.endswith('gz'):
            import gzip
            _open = gzip.open
        else:
            _open = open

        # parse log file
        with _open(file_path, 'rb') as f:
            xes_log = untangle.parse(f.read())

        # create event log
        event_log = EventLog()

        for trace_id, trace in enumerate(xes_log.log.trace):
            _trace = Trace(id=trace_id)
            for event in trace.event:
                event_name = event.string['value']
                event_timestamp = parser.parse(event.date['value'])
                _event = Event(name=event_name, start_time=event_timestamp)
                _trace.add_event(_event)
            event_log.add_trace(_trace)

        return event_log

    @staticmethod
    def from_csv(file_path):
        """
        Load an event log from a CSV file

        :param file_path: path to CSV file
        :return: EventLog object
        """
        # parse file as pandas dataframe
        df = pd.read_csv(file_path)

        # create event log
        event_log = EventLog()

        # iterate by distinct trace_id
        for trace_id in np.unique(df['trace_id']):
            _trace = Trace(id=trace_id)
            # iterate over rows per trace_id
            for index, row in df[df.trace_id == trace_id].iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                event_name = row['event']
                user = row['user']
                _event = Event(name=event_name, start_time=start_time, end_time=end_time, user=user)
                _trace.add_event(_event)
            event_log.add_trace(_trace)

        return event_log

    @staticmethod
    def from_csv(file_path):
        """
        Load an event log from a CSV file

        :param file_path: path to CSV file
        :return: EventLog object
        """
        # parse file as pandas dataframe
        df = pd.read_csv(file_path)

        # create event log
        event_log = EventLog()

        # iterate by distinct trace_id
        for trace_id in np.unique(df['trace_id']):
            _trace = Trace(id=trace_id)
            # iterate over rows per trace_id
            for index, row in df[df.trace_id == trace_id].iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                event_name = row['event']
                user = row['user']
                _event = Event(name=event_name, start_time=start_time, end_time=end_time, user=user)
                _trace.add_event(_event)
            event_log.add_trace(_trace)

        return event_log


class ProcessModel:
    def __init__(self, graph=None):
        """
        Init variables

        :param graph:
        """
        self.graph = graph
        self.start_event = START_EVENT_NAME
        self.end_event = END_EVENT_NAME
        self.variant_probabilities = None
        self.variants = None

    def load(self, file):
        """
        Load from a pickle file

        :param file:
        :return:
        """
        with open(file, 'rb') as f:
            self.graph = pickle.load(f)

    def save(self, file):
        """
        Save to a pickle file

        :param file:
        :return:
        """
        with open(file, 'wb') as f:
            pickle.dump(self.graph, f)

    def check_trace(self, trace):
        """
        Returns a list of booleans representing whether a transition within the trace is an anomaly or not.
        True = anomaly
        False = normal

        :param trace: Trace object
        :return: list of booleans
        """

        # zip(...) generates the edges from the traces
        return self.check_edges(zip(trace[:-1], trace[1:]))

    def check_traces(self, traces):
        """
        Returns a list of booleans for each trace. Cf. check_trace()

        :param traces: list of traces
        :return: list of list of booleans
        """
        return np.array([self.check_trace(s) for s in traces])

    def check_edge(self, edge):
        """
        Returns whether the edge is an anomaly or not.
        True = anomaly
        False = normal

        :param edge: edge
        :return: boolean
        """
        return edge in self.graph.edges()

    def check_edges(self, edges):
        """
        Returns for a list of given edges whether an edge is an anomaly. Cf. check_edge()

        :param edges: list of edges
        :return: list of booleans
        """
        return np.array([self.check_edge(e) for e in edges])

    def get_variants(self, probabilities=False, generate_users=False, generate_long_term_dependencies=False):
        """
        Return all possible variants from the process model.
        If probabilities is set to True the implicit probabilities similar to a random walk are returned as well

        :param probabilities: boolean, return the probabilities
        :param generate_users: boolean, generate user attribute
        :param generate_long_term_dependencies: boolean, generate long-term dependencies in each trace
        :return:
        """
        self.variants = EventLog()

        num_users = None

        if generate_users:
            num_users = np.random.randint(10, 30)
            users = np.arange(num_users)
            self.variants.attr['attr_dims'] = {
                'name': len(self.graph.node),
                'user': int(num_users) + 2
            }
            g = self.graph
            for key in g.node.keys():
                random_users = np.sort(
                    np.random.choice(users, np.random.randint(1, 5), replace=False)).tolist()
                random_users = [str(u) for u in random_users]
                if key in [self.start_event, self.end_event]:
                    random_users = None
                g.node[key]['_possible_users'] = random_users

        for path in sorted(nx.all_simple_paths(self.graph, source=self.start_event, target=self.end_event)):
            path = path[1:-1]  # remove BOS and EOS
            trace = Trace(label='normal')
            if generate_users:
                trace.attr["user_voc_size"] = num_users
            for event in path:
                trace.add_event(Event(name=event, start_time=None, **dict(self.graph.node[event].items())))
            self.variants.add_trace(trace)

        if generate_long_term_dependencies:
            # add long term dependencies to every variant
            # this means that each variant will have exactly one long term dependency where the user must
            # be the same for two events
            for variant in self.variants:
                random_events = np.sort(
                    np.random.choice(range(len(variant.events)), 2, replace=False))  # remove BOS and EOS
                random_attr = np.random.choice(list(variant[random_events[1]].attr.keys()), replace=False)
                head = random_events[0]
                tail = random_events[1:]

                for idx in tail:
                    variant[idx].attr[random_attr] = int(head)  # point to earlier event

        if not self.variant_probabilities and probabilities:
            self.variant_probabilities = []
            for variant in self.variants:
                p = np.product(
                    [1.0 / max(1.0, len([edge[1] for edge in self.graph.edges() if edge[0] == event.name])) for
                     event in
                     variant])
                self.variant_probabilities.append(p)

        if probabilities:
            return self.variants, self.variant_probabilities
        else:
            return self.variants

    def generate_event_log(self, size, anomalies=None, p=None, variant_probabilities=None,
                           seed=None, generate_users=False, generate_long_term_dependencies=False, variants=None):
        """
        Generates an event log.

        :param generate_long_term_dependencies:
        :param generate_users:
        :param size: number of traces
        :param anomalies: a list of anomalies that can be applied to the generated traces
        :param p: probability of the occurrence of an anomaly
        :param variant_probabilities: user set probability distribution for the variants
        :param seed: a random seed
        :return: EventLog
        """

        if seed is not None:
            np.random.seed(seed)

        # get variants
        if variants is None:
            variants, variant_probs = self.get_variants(probabilities=True,
                                                        generate_long_term_dependencies=generate_long_term_dependencies,
                                                        generate_users=generate_users)
        else:
            variants = variants

        # convert variant event log to list
        variants_list = list(variants)

        # set the variant probabilities
        if variant_probabilities is None:
            variant_probabilities = variant_probs

        start_date = datetime(2000, 1, 1).toordinal()
        end_date = datetime.today().toordinal()

        random_start_date = datetime.fromordinal(np.random.randint(start_date, end_date))
        duration_since_last_trace = 0
        trace_start_date = random_start_date

        # generate traces by randomly selecting a variant according to the given variant probability distribution
        traces = []
        for i in range(size):
            trace_start_date += timedelta(seconds=duration_since_last_trace)

            # select random variant
            variant = np.random.choice(variants_list, replace=True, p=variant_probabilities)

            # duration since last event
            duration_since_last_event = 0
            event_start_time = trace_start_date

            # init new trace and copy the attributes from the variant
            trace = Trace(trace_id=i, **variant.attr)
            for event in variant:
                event_start_time += timedelta(seconds=duration_since_last_event)
                event_end_time = event_start_time + timedelta(seconds=np.random.binomial(8 * 60 * 60, 0.1))

                # copy the events from the variant
                event_ = Event(name=event.name,
                               start_time=event_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                               end_time=event_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                               **event.attr)

                if generate_users:
                    u = event.attr["_possible_users"]
                    if isinstance(u, list):
                        event_.attr["user"] = np.random.choice(u)
                    elif isinstance(u, int):
                        event_.attr["user"] = trace[u].attr["user"]  # this is a long term dependency
                    else:
                        event_.attr["user"] = None

                trace.add_event(event_)

                # update duration since last event
                duration_since_last_event = np.random.binomial(7 * 24 * 60 * 60, 0.2)

            # apply anomalies according to the probability
            if anomalies:
                if np.random.uniform(0, 1) <= p:
                    anomaly = np.random.choice(anomalies)
                    trace = anomaly.apply(trace)

            # remove all users from trace as we don't need it anymore
            if generate_users:
                del trace.attr['user_voc_size']

            # append copied trace to the list of traces
            traces.append(trace)

            # update duration
            # duration is between 0 and 7 days normally distributed
            duration_since_last_trace = np.random.binomial(90 * 24 * 60 * 60, 0.2)

        # return
        event_log = EventLog(traces=traces, **variants.attr)
        return event_log

    @staticmethod
    def from_plg(file_path):
        """
        Load a process model from a plg file (the format the PLG2 tool uses)

        :param file_path: path to plg file
        :return: ProcessModel object
        """

        with open(file_path) as f:
            file_content = untangle.parse(f.read())

        start_event = int(file_content.process.elements.startEvent['id'])
        end_event = int(file_content.process.elements.endEvent['id'])

        id_activity = dict((int(task['id']), str(task['name'])) for task in file_content.process.elements.task)
        id_activity[start_event] = START_EVENT_NAME
        id_activity[end_event] = END_EVENT_NAME

        activities = id_activity.keys()

        gateways = [int(g['id']) for g in file_content.process.elements.gateway]
        gateway_followers = dict((id_, []) for id_ in gateways)
        followers = dict((id_, []) for id_ in activities)

        for sf in file_content.process.elements.sequenceFlow:
            source = int(sf['sourceRef'])
            target = int(sf['targetRef'])
            if source in gateways:
                gateway_followers[source].append(target)

        for sf in file_content.process.elements.sequenceFlow:
            source = int(sf['sourceRef'])
            target = int(sf['targetRef'])
            if source in activities and target in activities:
                followers[source].append(target)
            elif source in activities and target in gateways:
                followers[source] = gateway_followers.get(target)

        graph = nx.DiGraph()
        graph.add_nodes_from([id_activity.get(activity) for activity in activities])
        for source, targets in followers.items():
            for target in targets:
                graph.add_edge(id_activity.get(source), id_activity.get(target))

        return ProcessModel(graph)

    @staticmethod
    def from_plg_from_models_dir(model):
        """
        Load a plg file from the models dir under /.res/models

        :param model:
        :return:
        """
        if not model.endswith('.plg'):
            model = '.'.join((model, 'plg'))
        model_path = os.path.join(MODELS_DIR, model)
        return ProcessModel.from_plg(model_path)
