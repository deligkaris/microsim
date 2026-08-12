import unittest

import pandas as pd

from microsim.common.population_type import PopulationType
from microsim.common.variable_type import VariableType
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.person.person_filter import PersonFilter
from microsim.population.population_factory import PopulationFactory, VariablesUsedRecorder
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType

CAT = PopulationFactory.variable_types(VariableType.CATEGORICAL.value,
                                       popType=PopulationType.NHANES.value)
CON = PopulationFactory.variable_types(VariableType.CONTINUOUS.value,
                                       popType=PopulationType.NHANES.value)


def _group(gender, sbp=140., age=50., rows=2):
    """One group's dataframe: every row shares the categorical values, which is what lets a
       categorical-only filter be decided once for the whole group."""
    values = dict()
    for variable in CAT:
        values[variable] = [1] * rows
    values[StaticRiskFactorsType.GENDER.value] = [gender] * rows
    for variable in CON:
        values[variable] = [0.] * rows
    values[DynamicRiskFactorsType.SBP.value] = [sbp] * rows
    values[DynamicRiskFactorsType.AGE.value] = [age] * rows
    return pd.DataFrame(values)


def _dfForGroups():
    """Two groups differing only in gender, so a gender filter should keep exactly one."""
    return {("male",): _group(NHANESGender.MALE.value),
            ("female",): _group(NHANESGender.FEMALE.value)}


def _filter(filterName, filterFunction):
    pf = PersonFilter()
    pf.add_filter("df", filterName, filterFunction)
    return pf


class TestCategoricalFiltersDecideGroups(unittest.TestCase):
    """A filter that reads only categorical variables holds for every person in a group, so the
       whole group can be dropped before a distribution is fit for it."""

    def test_categorical_filter_drops_the_groups_it_rejects(self):
        pf = _filter("men", lambda x: x[StaticRiskFactorsType.GENDER.value] == NHANESGender.MALE.value)
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertEqual([("male",)], list(kept.keys()))

    def test_the_kept_group_dataframe_is_unchanged(self):
        pf = _filter("men", lambda x: x[StaticRiskFactorsType.GENDER.value] == NHANESGender.MALE.value)
        groups = _dfForGroups()
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(groups, pf)
        pd.testing.assert_frame_equal(groups[("male",)], kept[("male",)])

    def test_a_second_categorical_filter_also_applies(self):
        pf = _filter("men", lambda x: x[StaticRiskFactorsType.GENDER.value] == NHANESGender.MALE.value)
        pf.add_filter("df", "noStatin",
                      lambda x: x[DefaultTreatmentsType.STATIN.value] == 0)
        #every group in the fixture has statin==1, so nothing survives both filters
        with self.assertRaises(RuntimeError):
            PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)


class TestContinuousFiltersAreLeftAlone(unittest.TestCase):
    """A filter needing a continuous variable cannot be decided for a group: two people sharing the
       same categorical values can still fall on opposite sides of an SBP cut-off."""

    def test_continuous_filter_keeps_every_group(self):
        pf = _filter("lowSBPLimit", lambda x: x[DynamicRiskFactorsType.SBP.value] > 126)
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertEqual(2, len(kept))

    def test_continuous_filter_keeps_groups_it_would_reject_row_by_row(self):
        #sbp is 140 in the fixture, so this filter rejects every row, but it is still not decidable
        #from the categorical variables and must not drop a group here
        pf = _filter("impossibleSBP", lambda x: x[DynamicRiskFactorsType.SBP.value] > 1000)
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertEqual(2, len(kept))

    def test_mixed_and_filter_still_drops_what_the_categorical_half_rejects(self):
        #"and" short-circuits: for the female group the gender test is already False and the sbp test
        #never runs, so no KeyError is raised and the group is dropped -- correctly, since no woman
        #passes this filter whatever her sbp. The male group reaches the sbp test, raises, and stays.
        pf = _filter("menWithHighSBP",
                     lambda x: (x[StaticRiskFactorsType.GENDER.value] == NHANESGender.MALE.value)
                     and (x[DynamicRiskFactorsType.SBP.value] > 126))
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertEqual([("male",)], list(kept.keys()))

    def test_mixed_or_filter_keeps_every_group(self):
        #"or" does not short-circuit to False here: the female group falls through to the sbp test,
        #raises, and is left undecided, which is right because a woman with high sbp would pass
        pf = _filter("menOrHighSBP",
                     lambda x: (x[StaticRiskFactorsType.GENDER.value] == NHANESGender.MALE.value)
                     or (x[DynamicRiskFactorsType.SBP.value] > 126))
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertEqual(2, len(kept))


class TestVariablesUsedRecorder(unittest.TestCase):
    """The recorder is what makes a filter's dependencies visible, so that they can be compared against
       the categorical list in PopulationFactory.variable_types."""

    def test_records_what_was_read_and_returns_the_value(self):
        row = _group(NHANESGender.MALE.value, sbp=140.).iloc[0]
        recorder = VariablesUsedRecorder(row)
        self.assertEqual(NHANESGender.MALE.value, recorder[StaticRiskFactorsType.GENDER.value])
        self.assertEqual(140., recorder[DynamicRiskFactorsType.SBP.value])
        self.assertEqual({StaticRiskFactorsType.GENDER.value, DynamicRiskFactorsType.SBP.value},
                         recorder.variablesUsed)

    def test_records_nothing_when_nothing_is_read(self):
        recorder = VariablesUsedRecorder(_group(NHANESGender.MALE.value).iloc[0])
        self.assertEqual(set(), recorder.variablesUsed)

    def test_records_only_what_a_short_circuiting_filter_reached(self):
        #the gender test is False for this row, so the sbp test never runs and sbp is never recorded
        row = _group(NHANESGender.FEMALE.value).iloc[0]
        recorder = VariablesUsedRecorder(row)
        _ = ((recorder[StaticRiskFactorsType.GENDER.value] == NHANESGender.MALE.value)
             and (recorder[DynamicRiskFactorsType.SBP.value] > 126))
        self.assertEqual({StaticRiskFactorsType.GENDER.value}, recorder.variablesUsed)


class TestNoFiltersAndEmptyResult(unittest.TestCase):

    def test_none_returns_the_groups_unchanged(self):
        groups = _dfForGroups()
        self.assertIs(groups, PopulationFactory.apply_categorical_person_filters_on_groups(groups, None))

    def test_person_level_filters_are_ignored(self):
        pf = PersonFilter()
        pf.add_filter("person", "nobody", lambda x: False)
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertEqual(2, len(kept))

    def test_filter_on_a_variable_the_group_df_lacks_is_left_alone(self):
        #year is not one of the variables carried here, so the filter cannot be decided and must not drop
        pf = _filter("year1999", lambda x: x["year"] == 1999)
        kept = PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertEqual(2, len(kept))

    def test_rejecting_every_group_raises(self):
        pf = _filter("nobody", lambda x: x[StaticRiskFactorsType.GENDER.value] == -1)
        with self.assertRaises(RuntimeError) as cm:
            PopulationFactory.apply_categorical_person_filters_on_groups(_dfForGroups(), pf)
        self.assertIn("no distribution left to draw from", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
