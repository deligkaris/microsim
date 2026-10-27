"""Tests for the head counts of state populations (get_dataframe_with_categoricals).

Rows hold fractional expected counts that are rounded once, together, by systematic rounding, so
the total and every projection cell must come within 1 person of n*samplingRate. WY is the
smallest-cell case the old per-row rounding failed on (4,784 people instead of 6,363 at 0.01)."""

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from microsim.common.data_loader import get_absolute_datafile_path
from microsim.default_treatments.default_treatments import DEFAULT_TREATMENT_MIN_AGE
from microsim.population.population_factory import PopulationFactory
from microsim.validation import Validation

CELL = [
    "ageGroup",
    "raceEthnicity",
    "gender",
    "education",
    "smokingStatus",
    "anyPhysicalActivity",
]


class TestStatePopulationCounts(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        projection = pd.read_csv(
            get_absolute_datafile_path("state") + "/pop_projection_wy_2025.csv"
        )
        self.nForCell = projection.groupby(CELL)["n"].sum()

    def counts_for_cell(self, samplingRate):
        df = PopulationFactory.get_dataframe_with_categoricals(
            year=2025, state="WY", samplingRate=samplingRate
        )
        return df.groupby(CELL).size().reindex(self.nForCell.index, fill_value=0)

    def test_total_is_within_one_person(self):
        counts = self.counts_for_cell(0.01)
        self.assertLessEqual(abs(counts.sum() - self.nForCell.sum() * 0.01), 1)

    def test_every_cell_is_within_one_person(self):
        counts = self.counts_for_cell(0.01)
        self.assertLessEqual((counts - self.nForCell * 0.01).abs().max(), 1)

    def test_full_rate_keeps_the_whole_state(self):
        counts = self.counts_for_cell(1.0)
        self.assertLessEqual(abs(counts.sum() - self.nForCell.sum()), 1)

    def test_age_range_inverts_age_group(self):
        for age in range(120):
            first, last = PopulationFactory.get_age_range_from_ageGroup(
                PopulationFactory.get_ageGroup_from_age(age)
            )
            self.assertLessEqual(first, age)
            self.assertTrue(last is None or age <= last)
        self.assertEqual(PopulationFactory.get_age_range_from_ageGroup(1), (0, 4))
        self.assertEqual(PopulationFactory.get_age_range_from_ageGroup(17), (80, None))

    def test_every_gender_race_age_has_a_pooled_fit(self):
        keys = PopulationFactory.get_crude_distributions()["mean"].keys()
        for gender in [1, 2]:
            for raceEthnicity in [1, 2, 3, 4, 5]:
                for age in range(82):
                    self.assertIn((gender, raceEthnicity, age), keys)

    def test_nhanes_children_are_all_active(self):
        df = PopulationFactory.get_nhanesDf()
        minAge = PopulationFactory.ACTIVITY_MIN_GROUP_AGE
        self.assertTrue(df.loc[df["age"] < minAge, "anyPhysicalActivity"].astype(bool).all())
        self.assertFalse(df.loc[df["age"] == minAge, "anyPhysicalActivity"].astype(bool).all())

    def test_nhanes_young_are_untreated(self):
        df = PopulationFactory.get_nhanesDf()
        young = df[df["age"] < DEFAULT_TREATMENT_MIN_AGE]
        self.assertFalse(young["statin"].any())
        self.assertTrue((young["antiHypertensiveCount"] == 0).all())
        self.assertTrue(df.loc[df["age"] >= DEFAULT_TREATMENT_MIN_AGE, "statin"].any())

    def test_inactive_children_are_shifted(self):
        df = PopulationFactory.get_dataframe_with_categoricals(
            year=2025, state="WY", samplingRate=0.01
        )
        children = df[df["age"].astype(int) < PopulationFactory.ACTIVITY_MIN_GROUP_AGE]
        active = PopulationFactory.look_up_group_means(
            children.assign(anyPhysicalActivity=1), poolHispanics=True
        )
        inactive = PopulationFactory.look_up_group_means(
            children.assign(anyPhysicalActivity=0), poolHispanics=True
        )
        self.assertFalse(np.isnan(inactive).any())
        np.testing.assert_array_equal(inactive, active)

    def test_state_with_groups_nhanes_lacks_draws(self):
        # IA holds men of race other with less than high school aged 25-29, whom NHANES lacks
        df = PopulationFactory.get_dataframe_with_categoricals(
            year=2025, state="IA", samplingRate=0.01
        )
        df = PopulationFactory.append_dataframe_with_continuous(
            df, PopulationFactory.get_crude_distributions(poolHispanics=True), poolHispanics=True
        )
        drawn = PopulationFactory.continuous_variables_drawn()
        self.assertFalse(df[drawn].isna().any().any())

    def test_built_population_matches_projection(self):
        # the built Persons, not just the dataframe they are built from
        with contextlib.redirect_stdout(io.StringIO()):
            results = Validation.state_baseline_pop("WY", samplingRate=0.005)
        self.assertLessEqual(abs(results["n"] - results["expectedN"]), 1)
        self.assertLessEqual(results["maxCellError"], 1)
        # not bounded by the rounding like the cells: across 500 seeds at this size the worst
        # was 0.0046 (smokingStatus), under the ~0.007 sd of a simple random sample
        for marginal in results["marginals"].values():
            self.assertLess(marginal["difference"].abs().max(), 0.01)

    def test_report_prints_every_category(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = Validation.state_baseline_pop("WY", samplingRate=0.002)
        report = out.getvalue()
        for variable, marginal in result["marginals"].items():
            self.assertIn(f"\n  {variable}\n", report)
            for code in marginal.index:
                self.assertIn(f"    {Validation._state_category_label(variable, code)} ", report)


class TestStateHispanicsPooled(unittest.TestCase):
    """The projections code every Hispanic as Other Hispanic (2), so the state pipeline reads
    NHANES with Mexican Americans (1) recoded to 2; NHANES populations keep the two apart."""

    def test_fits_pool_hispanics_and_keep_other_races(self):
        pooled = PopulationFactory.get_crude_distributions(poolHispanics=True)["mean"]
        unpooled = PopulationFactory.get_crude_distributions()["mean"]
        self.assertFalse(any(key[1] == 1 for key in pooled))
        self.assertTrue(any(key[1] == 1 for key in unpooled))
        for gender in [1, 2]:
            for raceEthnicity in [2, 3, 4, 5]:
                for age in range(82):
                    self.assertIn((gender, raceEthnicity, age), pooled)
        # the pooled fits recode the same bootstrap, so the other races fit the same rows
        for key in unpooled:
            if key[1] > 2:
                np.testing.assert_array_equal(unpooled[key], pooled[key])

    def test_group_means_pool_hispanics(self):
        df = PopulationFactory.get_nhanesDf_resampled()
        keys = PopulationFactory.group_key_frame(df)
        hispanic = keys["raceEthnicity"].isin([1, 2])
        others = keys.drop(columns="raceEthnicity")
        # the group holding the most Mexican Americans, which holds Other Hispanics too
        group = others[keys["raceEthnicity"] == 1].value_counts().index[0]
        inGroup = hispanic & (others == group).all(axis=1)
        self.assertGreater((keys.loc[inGroup, "raceEthnicity"] == 2).sum(), 0)
        expected = df.loc[inGroup, PopulationFactory.continuous_variables_drawn()].mean()
        pooled = PopulationFactory.get_group_means(poolHispanics=True)
        key = (group[0], 2, *group[1:])
        np.testing.assert_allclose(pooled.loc[key].to_numpy(), expected.to_numpy())
        self.assertNotIn(1, pooled.index.get_level_values("raceEthnicity"))
        self.assertIn(
            1, PopulationFactory.get_group_means().index.get_level_values("raceEthnicity")
        )

    def test_treatment_proportions_pool_hispanics(self):
        df = PopulationFactory.get_nhanesDf()
        ageGroup = df["age"].apply(PopulationFactory.get_ageGroup_from_age)
        rows = df[(ageGroup == 12) & (df["gender"] == 2) & df["raceEthnicity"].isin([1, 2])]
        onBoth = rows["statin"] & (rows["antiHypertensiveCount"] == 1)
        expected = rows.loc[onBoth, "WTINT2YR"].sum() / rows["WTINT2YR"].sum()
        proportions = PopulationFactory.get_proportionForDefaultTreatments()
        self.assertAlmostEqual(expected, proportions[12, 2, 2][True, 1.0])
        self.assertNotIn((12, 2, 1), proportions)


if __name__ == "__main__":
    unittest.main()
