from microsim.store.numpy_population_iterator import NumpyPopulationIterator
import numpy as np


class NumpyPopulationProxy:
    def __init__(
        self,
        static_rows,
        dynamic_rows,
        event_rows,
        new_person_record_proxy,
        active_indices=None,
        active_condition=None,
    ):
        self._static_rows = static_rows
        self._dynamic_rows = dynamic_rows
        self._event_rows = event_rows
        self._new_person_record_proxy = new_person_record_proxy

        if active_indices is not None:
            self._active_indices = active_indices
        elif active_condition is not None:
            active_mask = self._unconditional_apply(active_condition, out_dtype=np.bool8)
            (self._active_indices,) = active_mask.nonzero()
        else:
            raise ValueError("Expected to receive one of: `active_indices`, `active_condition`")

    @property
    def active_indices(self):
        return self._active_indices

    @property
    def num_persons(self):
        return self._active_indices.shape[0]

    def __iter__(self):
        return NumpyPopulationIterator(
            self._static_rows,
            self._dynamic_rows,
            self._event_rows,
            self._active_indices,
            self._new_person_record_proxy,
        )

    def _unconditional_apply(self, func, out_dtype=np.float64, **kwargs):
        """
        Applies `func` to each person record, then returns the result.

        Runs `func` even if person record is not active (i.e., even if its
        index does not appear in `self._active_indices`). Necessary to set
        `self._active_indices` in __init__ if given a condition, but may have
        other uses.
        """
        ops = [self._static_rows, self._dynamic_rows, self._event_rows, None]
        flags = []
        op_flags = [["readonly"], ["readonly"], ["readonly"], ["writeonly", "allocate"]]
        op_dtypes = [
            self._static_rows.dtype,
            self._dynamic_rows.dtype,
            self._event_rows.dtype,
            out_dtype,
        ]
        with np.nditer(ops, flags, op_flags, op_dtypes) as it:
            for s, d, e, out in it:
                record_proxy = self._new_person_record_proxy(s, d, e)
                out[...] = func(record_proxy, **kwargs)
            return it.operands[3]
