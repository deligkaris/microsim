import numpy as np


class NumpyPopulationIterator:
    def __init__(self, person_store, at_t, active_indices):
        self._person_store = person_store
        self._at_t = at_t
        self._it = np.nditer(active_indices, [], ["readonly"], [active_indices.dtype])

    def __iter__(self):
        return self

    def __next__(self):
        abs_person_idx = next(self._it)
        person_proxy = self._person_store.get_person_proxy_at(abs_person_idx, self._at_t)
        return person_proxy