def is_alive(person_record):
    return person_record.alive


class StorePopulation:
    """
    Population that uses a PersonStore to store its people.

    Primarily intended for storing Person data in Numpy ndarrays for memory
    efficiency while still having Person-like objects for advancing, updating,
    and analyzing.
    """

    def __init__(self, person_store):
        self._person_store = person_store
        self._current_tick = 0

    @property
    def person_store(self):
        return self._person_store

    @property
    def current_tick(self):
        return self._current_tick

    def advance(self, num_ticks=1):
        """Advance population by a given number of ticks (default: 1)."""
        start_tick = self._current_tick + 1
        end_tick = start_tick + num_ticks
        for t in range(start_tick, end_tick + 1):
            cur_pop, next_pop = self._person_store.get_population_advance_record_window(
                t, condition=is_alive
            )

            if cur_pop.num_persons == 0:
                break

            # TODO: the rest of `advance` goes here

        self._current_tick = end_tick
