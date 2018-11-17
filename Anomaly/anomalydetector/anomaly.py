import numpy as np

from anomalydetector.processmining import Event


class Anomaly:
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)

    def __str__(self):
        return str(self.__class__.__name__)

    def apply(self, trace):
        """
        This method applies the anomaly to a given trace

        :param trace: the input trace
        :return: a new trace after the anomaly has been applied
        """
        pass


class MissingHead(Anomaly):
    def __init__(self, seed=None, max_sequence_size=3):
        self.max_sequence_size = max_sequence_size
        super().__init__(seed=seed)

    def apply(self, trace):
        size = self.random.randint(1, min(len(trace), self.max_sequence_size + 1))

        t = trace.events

        label = {
            "anomaly": __class__.__name__,
            "attr": {
                "size": int(size),
                "head": [event.name for event in t[:size]]
            }
        }

        anomalous_trace = t[size:]
        trace.events = anomalous_trace
        trace.attr["label"] = label
        return trace


class MissingTail(Anomaly):
    def __init__(self, seed=None, max_sequence_size=3):
        self.max_sequence_size = max_sequence_size
        super().__init__(seed=seed)

    def apply(self, trace):
        size = self.random.randint(1, min(len(trace), self.max_sequence_size + 1))

        t = trace.events

        label = {
            "anomaly": __class__.__name__,
            "attr": {
                "size": int(size),
                "tail": [event.name for event in t[-size:]]
            }
        }

        anomalous_trace = t[:-size]
        trace.events = anomalous_trace
        trace.attr["label"] = label
        return trace


class DuplicateSequence(Anomaly):
    def __init__(self, seed=None, max_sequence_size=1):
        self.max_sequence_size = max_sequence_size
        super().__init__(seed=seed)

    def apply(self, trace):
        sequence_size = self.random.randint(1, min(len(trace), self.max_sequence_size + 1))
        start = self.random.randint(0, len(trace) - sequence_size)

        label = {
            "anomaly": __class__.__name__,
            "attr": {
                "size": int(sequence_size),
                "start": int(start)
            }
        }

        t = trace.events
        dupe_event = t[start]
        dupe_event = Event(name=dupe_event.name, start_time=dupe_event.start_time, end_time=dupe_event.end_time,
                           **dupe_event.attr)

        # # set a new random user for the duplicated event
        # user = np.random.choice(dupe_event.attr['_possible_users'])
        # dupe_event.attr['user'] = user

        anomalous_trace = t[:start] + [dupe_event] + t[start:]
        trace.events = anomalous_trace
        trace.attr["label"] = label

        return trace


class IncorrectAttribute(Anomaly):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def apply(self, trace):
        long_term = [isinstance(event.attr["_possible_users"], int) for event in trace]
        long_term_indices = np.where(long_term)[0]
        possible_indices = set(range(len(trace))) - set(long_term_indices)
        index = self.random.choice(list(possible_indices), 1)[0]

        user_voc_size = set(map(str, range(trace.attr["user_voc_size"]))) - set(trace[index].attr["_possible_users"])
        original_user = trace[index].attr["user"]
        trace[index].attr["user"] = str(self.random.choice(list(user_voc_size)))
        trace.attr["label"] = {
            "anomaly": __class__.__name__,
            "attr": {
                "index": int(index),
                "affected": "user",
                "original_user": original_user
            }
        }
        return trace


class IncorrectLongTermDependency(Anomaly):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def apply(self, trace):
        long_term = [isinstance(event.attr["_possible_users"], int) for event in trace]
        if not any(long_term):
            return trace
        else:
            long_term_indexes = np.where(long_term)[0]
            for index in long_term_indexes:
                user_voc_size = list(range(trace.attr["user_voc_size"]))
                del user_voc_size[int(trace[index].attr["user"])]
                original_user = trace[index].attr["user"]
                trace[index].attr["user"] = str(self.random.choice(user_voc_size))
                trace.attr["label"] = {
                    "anomaly": __class__.__name__,
                    "attr": {
                        "index": int(index),
                        "affected": "user",
                        "original_user": original_user,
                    }
                }
            return trace


class SkipSequence(Anomaly):
    def __init__(self, seed=None, max_sequence_size=1):
        self.max_sequence_size = max_sequence_size
        super().__init__(seed=seed)

    def apply(self, trace):
        sequence_size = self.random.randint(1, min(len(trace), self.max_sequence_size + 1))
        start = self.random.randint(0, len(trace) - sequence_size)
        end = start + sequence_size

        label = {
            "anomaly": __class__.__name__,
            "attr": {
                "size": int(sequence_size),
                "start": int(start),
                "skipped_event": trace[start].to_json()
            }
        }

        t = trace.events
        anomalous_trace = t[:start] + t[end:]
        trace.events = anomalous_trace
        trace.attr["label"] = label
        return trace


class SwitchEvents(Anomaly):
    def __init__(self, seed=None, max_distance=1):
        self.max_distance = max_distance
        super().__init__(seed=seed)

    def apply(self, trace):
        distance = self.random.randint(1, min(len(trace) - 1, self.max_distance + 1))

        first = self.random.randint(0, len(trace) - 1 - distance)
        second = first + distance

        label = {
            "anomaly": __class__.__name__,
            "attr": {
                "first": int(first),
                "second": int(second)
            }
        }

        t = trace.events
        anomalous_trace = t[:first] + [t[second]] + t[first + 1:second] + [t[first]] + t[second + 1:]
        trace.events = anomalous_trace
        trace.attr["label"] = label
        return trace
