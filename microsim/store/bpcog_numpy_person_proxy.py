from itertools import chain
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus


class DirectProxiedProperty:
    def __init__(self, proxied_prop_name):
        self._proxied_prop_name = proxied_prop_name

    def __get__(self, instance, owner=None):
        cur_record = instance.current
        value = getattr(cur_record, self._proxied_prop_name)
        return value


class MeanProxiedProperty:
    def __init__(self, proxied_prop_name):
        self._proxied_prop_name = proxied_prop_name

    def __get__(self, instance, owner=None):
        cur_prev_records = instance.current_and_previous
        values = [getattr(r, self._proxied_prop_name) for r in cur_prev_records]
        mean = sum(values) / len(cur_prev_records)
        return mean


class HadPriorEventProxiedProperty:
    def __init__(self, proxied_prop_name):
        self._proxied_prop_name = proxied_prop_name

    def __get__(self, instance, owner=None):
        cur_prev_records = instance.current_and_previous
        had_prior_event = any(
            getattr(r, self._proxied_prop_name) is not None for r in cur_prev_records
        )
        return had_prior_event


class AgeAtFirstEventProxiedProperty:
    def __init__(self, event_prop_name):
        self._event_prop_name = event_prop_name

    def __get__(self, instance, owner=None):
        for record in chain(instance.current_and_previous, [instance.next]):
            event = getattr(record, self._event_prop_name)
            if event is not None:
                age = getattr(event, "age", None) or record.age
                return age
        return None


def new_bpcog_person_proxy_class(field_metadata, name="BPCOGNumpyPersonProxy"):
    # proxy latest properties for vectorized model compatibility
    all_record_prop_names = set(chain(*[c.keys() for c in field_metadata.values()]))

    def person_proxy_init(self, next_record, cur_prev_records):
        self._next_record = next_record
        self._cur_prev_records = list(cur_prev_records)

    base_attrs = {
        "__init__": person_proxy_init,
        "current": property(lambda self: self._cur_prev_records[-1]),
        "current_and_previous": property(lambda self: self._cur_prev_records),
        "next": property(lambda self: self._next_record),
    }

    prop_attrs = {}
    for prop_name in all_record_prop_names:
        prop_attrs[prop_name] = DirectProxiedProperty(prop_name)

    # add mean{prop_name} property for numeric dynamic fields
    for dynamic_prop_name in field_metadata["dynamic"].keys():
        mean_prop_name = f"mean{dynamic_prop_name.capitalize()}"
        prop_attrs[mean_prop_name] = MeanProxiedProperty(dynamic_prop_name)

    # add ([has_prior_]|ageAtFirst){prop_name} properties for events
    for event_prop_name in field_metadata["event"].keys():
        prop_attrs[event_prop_name] = HadPriorEventProxiedProperty(event_prop_name)

        cap_event_prop_name = "MI" if event_prop_name == "mi" else event_prop_name.capitalize()
        age_at_first_event_prop_name = f"ageAtFirst{cap_event_prop_name}"
        prop_attrs[age_at_first_event_prop_name] = AgeAtFirstEventProxiedProperty(event_prop_name)

    # boolean props for model args that currently cannot be specified easily or readably
    # ... for the CV outcome model
    prop_attrs["black"] = property(
        lambda p: p.current.raceEthnicity > NHANESRaceEthnicity.NON_HISPANIC_BLACK
    )
    prop_attrs["current_smoker"] = property(
        lambda p: p.current.smokingStatus == SmokingStatus.CURRENT
    )
    prop_attrs["current_diabetes"] = property(lambda p: p.current.a1c > 6.5)
    prop_attrs["current_bp_treatment"] = property(lambda p: p.current.antiHypertensiveCount > 0)
    # ... for the GCP model
    prop_attrs["baseAge"] = property(lambda p: p.current_and_previous[0].age)
    prop_attrs["totalYearsInSim"] = property(lambda p: p.current.age - p.baseAge)
    # ... and for the dementia model
    prop_attrs["baseGcp"] = property(lambda p: p.current_and_previous[0].gcp)
    prop_attrs["gcpSlope"] = property(
        lambda p: p.current.gcp - p.current_and_previous[-2].gcp
        if len(p.current_and_previous) > 1
        else 0
    )

    proxy_class_attrs = {**prop_attrs, **base_attrs}
    person_proxy_class = type(name, tuple(), proxy_class_attrs)
    return person_proxy_class