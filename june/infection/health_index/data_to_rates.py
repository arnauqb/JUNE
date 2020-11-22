import logging
import pandas as pd
import numpy as np
from typing import List
from june import paths
from scipy import interpolate, curve_fit

default_seroprevalence_file = (
    paths.data_path / "input/health_index/seroprevalence_by_age_imperial.csv"
)
default_care_home_seroprevalence_file = (
    paths.data_path / "input/health_index/care_home_seroprevalence_by_age.csv"
)
default_population_file = (
    paths.data_path
    / "input/health_index/corrected_population_by_age_sex_2020_england.csv"
)
default_all_deaths_file = (
    paths.data_path / "input/health_index/deaths_by_age_sex_17_july.csv"
)
default_care_home_deaths_file = (
    paths.data_path / "input/health_index/care_home_deaths_by_age_sex_17_july.csv"
)
default_care_home_ratios_by_age_sex_file = (
    paths.data_path / "input/health_index/care_home_ratios_by_age_sex_england.csv"
)
default_hospital_deaths_by_age_sex_file = (
    paths.data_path / "input/health_index/cocin_hospital_deaths_by_age_sex_8july.csv"
)
default_hospital_admissions_by_age_sex_file = (
    paths.data_path
    / "input/health_index/cocin_hospital_admissions_by_age_sex_8july.csv"
)

logger = logging.getLogger("rates")


def convert_to_intervals(ages: List[str]) -> pd.IntervalIndex:
    idx = []
    for age in ages:
        idx.append(
            pd.Interval(
                left=int(age.split("-")[0]), right=int(age.split("-")[1]), closed="both"
            )
        )
    return pd.IntervalIndex(idx)


def check_age_intervals(df: pd.DataFrame):
    age_intervals = list(df.index)
    lower_age = age_intervals[0].left
    upper_age = age_intervals[-1].right
    if lower_age != 0:
        logger.warning(
            f"Your age intervals do not contain values smaller than {lower_age}."
            f"We will presume ages from 0 to {lower_age} all have the same value."
        )
        age_intervals[0] = pd.Interval(
            left=0, right=age_intervals[0].right, closed="both"
        )
    if upper_age < 99:
        logger.warning(
            f"Your age intervals do not contain values larger than {upper_age}."
            f"We will presume ages {upper_age} all have the same value."
        )
        age_intervals[-1] = pd.Interval(
            left=age_intervals[-1].left, right=99, closed="both"
        )
    elif upper_age > 99:
        logger.warning(
            f"Your age intervals contain values larger than 99."
            f"Setting that to the be the uper limit"
        )
        age_intervals[-1] = pd.Interval(
            left=age_intervals[-1].left, right=99, closed="both"
        )
    df.index = age_intervals
    return df


def weighted_interpolation(value, weights):
    weights = np.array(weights)
    return weights * value / weights.sum()


class Data2Rates:
    def __init__(
        self,
        seroprevalence_df: pd.DataFrame,
        population_by_age_sex_df: pd.DataFrame,
        all_deaths_by_age_sex_df: pd.DataFrame,
        hospital_deaths_by_age_sex_df: pd.DataFrame,
        hospital_admissions_by_age_sex_df: pd.DataFrame,
        care_home_deaths_by_age_sex_df: pd.DataFrame = None,
        care_home_ratios_by_age_sex_df: pd.DataFrame = None,
        care_home_seroprevalence_by_age_df: pd.DataFrame = None,
    ):
        # seroprev
        self.seroprevalence_df = self._process_df(seroprevalence_df, converters=True)
        self.care_home_seroprevalence_by_age_df = self._process_df(
            care_home_seroprevalence_by_age_df, converters=True
        )

        # populations
        self.population_by_age_sex_df = self._process_df(
            population_by_age_sex_df, converters=False
        )
        self.care_home_ratios_by_age_sex_df = self._process_df(
            care_home_ratios_by_age_sex_df, converters=False
        )
        self.all_deaths_by_age_sex_df = self._process_df(
            all_deaths_by_age_sex_df, converters=True
        )
        self.care_home_deaths_by_age_sex_df = self._process_df(
            care_home_deaths_by_age_sex_df, converters=True
        )
        self.hospital_deaths_by_age_sex_df = self._process_df(
            hospital_deaths_by_age_sex_df, converters=True
        )
        self.hospital_admissions_by_age_sex_df = self._process_df(
            hospital_admissions_by_age_sex_df, converters=True
        )

    @classmethod
    def from_files(
        cls,
        seroprevalence_file: str = default_seroprevalence_file,
        population_file: str = default_population_file,
        all_deaths_file: str = default_all_deaths_file,
        hospital_deaths_file: str = default_hospital_deaths_by_age_sex_file,
        hospital_admissions_file: str = default_hospital_admissions_by_age_sex_file,
        care_home_deaths_file: str = default_care_home_deaths_file,
        care_home_ratios_by_age_sex_file: str = default_care_home_ratios_by_age_sex_file,
        care_home_seroprevalence_by_age_file: str = default_care_home_seroprevalence_file,
    ) -> "Data2Rates":

        seroprevalence_df = cls._read_csv(seroprevalence_file)
        population_df = cls._read_csv(population_file)
        all_deaths_df = cls._read_csv(all_deaths_file)
        hospital_deaths_df = cls._read_csv(hospital_deaths_file)
        hospital_admissions_df = cls._read_csv(hospital_admissions_file)
        if care_home_deaths_file is None:
            care_home_deaths_df = None
        else:
            care_home_deaths_df = cls._read_csv(care_home_deaths_file)
        if care_home_ratios_by_age_sex_file is None:
            care_home_ratios_by_age_sex_df = None
        else:
            care_home_ratios_by_age_sex_df = cls._read_csv(
                care_home_ratios_by_age_sex_file
            )
        if care_home_seroprevalence_by_age_file is None:
            care_home_seroprevalence_by_age_df = None
        else:
            care_home_seroprevalence_by_age_df = cls._read_csv(
                care_home_seroprevalence_by_age_file
            )
        return cls(
            seroprevalence_df=seroprevalence_df,
            population_by_age_sex_df=population_df,
            all_deaths_by_age_sex_df=all_deaths_df,
            hospital_deaths_by_age_sex_df=hospital_deaths_df,
            hospital_admissions_by_age_sex_df=hospital_admissions_df,
            care_home_deaths_by_age_sex_df=care_home_deaths_df,
            care_home_ratios_by_age_sex_df=care_home_ratios_by_age_sex_df,
            care_home_seroprevalence_by_age_df=care_home_seroprevalence_by_age_df,
        )

    @classmethod
    def _read_csv(cls, filename):
        df = pd.read_csv(filename)
        df.set_index("age", inplace=True)
        return df

    def _process_df(self, df, converters=True, interpolate_bins=False):
        if converters:
            new_index = convert_to_intervals(df.index)
            df.index = new_index
            df = check_age_intervals(df=df)
        if interpolate_bins:
            df = self._interpolate_bins(df=df)
        return df

    def _get_interpolated_value(self, df, age, sex, weight_mapper=None):
        """
        Interpolates bins to single years by weighting each year by its population times
        the death rate

        Parameters
        ----------
        df
            dataframe with the structure
            age | male | female
            0-5 | 2    | 3
            etc.
       weight_mapper 
            function mapping (age,sex) -> weight
            if not provided uses population weight.
        """
        if weight_mapper is None:
            weight_mapper = lambda age, sex: self.population_by_age_sex_df.loc[age, sex]
        age_bin = df.loc[age].name
        data_bin = df.loc[age, sex]
        bin_weight = sum(
            [
                weight_mapper(age_i, sex)
                for age_i in range(age_bin.left, age_bin.right + 1)
            ]
        )
        value_weight = weight_mapper(age, sex)
        return value_weight * data_bin / bin_weight

    def get_n_cases(
        self, age: int, sex: str, is_care_home: bool = False, weight_mapper=None
    ) -> float:
        if is_care_home:
            sero_prevalence = self.care_home_seroprevalence_by_age_df.loc[
                age, "seroprevalence"
            ]
            n_people = self.population_by_age_sex_df.loc[age, sex]
            n_people *= self.care_home_ratios_by_age_sex_df.loc[age, sex]

        else:
            sero_prevalence = self.seroprevalence_df.loc[age, "seroprevalence"]
            n_people = self.population_by_age_sex_df.loc[age, sex]
            if self.care_home_ratios_by_age_sex_df is not None:
                n_people -= n_people * self.care_home_ratios_by_age_sex_df.loc[age, sex]
        n_cases = n_people * sero_prevalence
        # correct for deaths
        n_cases += self.get_n_deaths(
            age=age, sex=sex, is_care_home=is_care_home, weight_mapper=weight_mapper
        )
        return n_cases

    def get_care_home_deaths(self, age: int, sex: str, weight_mapper=None):
        pop_mapper = (
            lambda age, sex: self.population_by_age_sex_df.loc[age, sex]
             * self.care_home_ratios_by_age_sex_df.loc[age, sex]
        )
        if weight_mapper is None:
            mapper = pop_mapper
        else:
            mapper = lambda age, sex: weight_mapper(age, sex) * pop_mapper(age, sex)
        return self._get_interpolated_value(
            df=self.care_home_deaths_by_age_sex_df,
            age=age,
            sex=sex,
            weight_mapper=mapper,
        )

    def get_all_deaths(self, age: int, sex: str, weight_mapper=None):
        pop_mapper = lambda age, sex: self.population_by_age_sex_df.loc[age, sex]
        if weight_mapper is None:
            mapper = pop_mapper
        else:
            mapper = lambda age, sex: weight_mapper(age, sex) * pop_mapper(age, sex)
        return self._get_interpolated_value(
            df=self.all_deaths_by_age_sex_df, age=age, sex=sex, weight_mapper=mapper
        )

    def get_n_deaths(
        self, age: int, sex: str, is_care_home: bool = False, weight_mapper=None
    ) -> int:
        if is_care_home:
            return self.get_care_home_deaths(
                age=age, sex=sex, weight_mapper=weight_mapper
            )
        else:
            deaths_total = self.get_all_deaths(
                age=age, sex=sex, weight_mapper=weight_mapper
            )
            if self.care_home_deaths_by_age_sex_df is None:
                return deaths_total
            else:
                deaths_care_home = self.get_care_home_deaths(
                    age=age, sex=sex, weight_mapper=weight_mapper
                )
                return deaths_total - deaths_care_home

    def get_infection_fatality_rate(
        self, age: int, sex: str, is_care_home: bool = False, weight_mapper=None
    ) -> float:
        n_cases = self.get_n_cases(
            age=age, sex=sex, is_care_home=is_care_home, weight_mapper=weight_mapper
        )
        n_deaths = self.get_n_deaths(
            age=age, sex=sex, is_care_home=is_care_home, weight_mapper=weight_mapper
        )
        if n_cases == n_deaths:
            return 0
        else:
            return n_deaths / n_cases

    def get_n_hospital_deaths(self, age: int, sex: str, weight_mapper=None) -> int:
        pop_mapper = lambda age, sex: self.population_by_age_sex_df.loc[age, sex]
        if weight_mapper is None:
            mapper = pop_mapper
        else:
            mapper = lambda age, sex: weight_mapper(age, sex) * pop_mapper(age, sex)
        return self._get_interpolated_value(
            df=self.hospital_deaths_by_age_sex_df,
            age=age,
            sex=sex,
            weight_mapper=mapper,
        )

    def get_n_hospital_admissions(self, age: int, sex: str, weight_mapper=None) -> int:
        pop_mapper = lambda age, sex: self.population_by_age_sex_df.loc[age, sex]
        if weight_mapper is None:
            mapper = pop_mapper
        else:
            mapper = lambda age, sex: weight_mapper(age, sex) * pop_mapper(age, sex)
        return self._get_interpolated_value(
            df=self.hospital_admissions_by_age_sex_df,
            age=age,
            sex=sex,
            weight_mapper=mapper,
        )

    def get_hospital_death_rate(self, age: int, sex: str, weight_mapper=None) -> int:
        return self.get_n_hospital_deaths(
            age=age, sex=sex, weight_mapper=weight_mapper
        ) / self.get_n_hospital_admissions(
            age=age, sex=sex, weight_mapper=weight_mapper
        )

    def get_hospital_infection_fatality_rate(
        self, age: int, sex: str, weight_mapper=None
    ) -> int:
        n_cases = self.get_n_cases(
            age=age, sex=sex, is_care_home=False, weight_mapper=weight_mapper
        )
        n_hospital_deaths = self.get_n_hospital_deaths(
            age=age, sex=sex, weight_mapper=weight_mapper
        )
        return n_hospital_deaths / n_cases

    def get_functional_infection_fataility_rate(self, death_rate_df, bin_size=5):
        age_lows = np.arange(0, 100, bin_size)
        age_highs = np.arange(bin_size - 1, 100, bin_size)
        age_mids = (age_highs - age_lows) / 2
        male_drs = []
        female_drs = []
        for age_low, age_high in zip(age_lows, age_highs):
            death_rate_males = [
                death_rate_df.loc[age, "male"] for age in range(age_low, age_high + 1)
            ]
            death_rate_females = [
                death_rate_df.loc[age, "female"] for age in range(age_low, age_high + 1)
            ]
            death_rate_avg_males = np.average(
                death_rate_males,
                weights=self.population_by_age_sex_df.loc[
                    age_low:age_high, "male"
                ].values,
            )
            death_rate_avg_females = np.average(
                death_rate_females,
                weights=self.population_by_age_sex_df.loc[
                    age_low:age_high, "female"
                ].values,
            )
            male_drs.append(death_rate_avg_males)
            female_drs.append(death_rate_avg_females)
        deg = len(age_mids)
        male_fit = np.polyfit(x=age_mids, y=np.log(male_drs), deg=deg)
        female_fit = np.polyfit(x=age_mids, y=np.log(female_drs), deg=deg)
        male_func = lambda age: np.exp(
            sum(coef * age ** (deg - i) for i, coef in enumerate(male_fit))
        )
        female_func = lambda age: np.exp(
            sum(coef * age ** (deg - i) for i, coef in enumerate(female_fit))
        )
        return male_func, female_func


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rates = Data2Rates.from_files()
    # iteration loop
    # age_range = range(0, 100)
    # age_range_bins = np.arange(0,100,5)
    # death_rate_df = pd.DataFrame(index=age_range)
    # death_rate_df["male"] = [
    #    rates.get_infection_fatality_rate(age=age, sex="male") for age in age_range
    # ]
    # death_rate_df["female"] = [
    #    rates.get_infection_fatality_rate(age=age, sex="female") for age in age_range
    # ]
    # male_func, female_func = rates.get_functional_infection_fataility_rate(
    #    death_rate_df=death_rate_df
    # )
    # fig, ax = plt.subplots()
    # death_rate_df.plot(ax=ax)
    # plt.plot(age_range, [male_func(age) for age in age_range], label="fit male")
    # plt.plot(age_range, [female_func(age) for age in age_range], label="fit female")
    # plt.legend()
    # plt.show()
    # death_rate_df.plot()
    # plt.show()
    # n_iterations = 5
    # for i in range(n_iterations):
    #    death_rate_df_refined = pd.DataFrame(index=age_range)
    #    weight_mapper = lambda age, sex: death_rate_df.loc[age, sex]
    #    death_rate_df_refined["male"] = [
    #        rates.get_infection_fatality_rate(
    #            age=age, sex="male", weight_mapper=weight_mapper
    #        )
    #        for age in age_range
    #    ]
    #    death_rate_df_refined["female"] = [
    #        rates.get_infection_fatality_rate(
    #            age=age, sex="female", weight_mapper=weight_mapper
    #        )
    #        for age in age_range
    #    ]
    #    death_rate_df_refined.plot()
    #    plt.show()
    #    death_rate_df = death_rate_df_refined

    ifr_by_sex_ward = pd.DataFrame.from_dict(
        {"male": 1.07, "female": 0.71}, orient="index"
    )
    all_male_deaths, all_male_cases = 0, 0
    all_female_deaths, all_female_cases = 0, 0
    for age in np.arange(0, 100):
        all_male_deaths += rates.get_n_deaths(age=age, sex="male")
        all_male_cases += rates.get_n_cases(age=age, sex="male")
        all_female_deaths += rates.get_n_deaths(age=age, sex="female")
        all_female_cases += rates.get_n_cases(age=age, sex="female")
    overall_male_dr = all_male_deaths / all_male_cases * 100
    overall_female_dr = all_female_deaths / all_female_cases * 100
    ifr_by_sex = pd.DataFrame.from_dict(
        {"male": overall_male_dr, "female": overall_female_dr}, orient="index"
    )
    fig, ax = plt.subplots()
    ifr_by_sex_ward[0].plot.bar(ax=ax, label="Ward et al", alpha=0.3, color="blue")
    ifr_by_sex[0].plot.bar(ax=ax, label="JUNE", alpha=0.3, color="orange")
    plt.ylabel("IFR by sex")
    plt.legend()
    plt.show()
    # ***************** Ward et al IFR
    ifr_age = [0.03, 0.52, 3.13, 11.64]
    ages = [
        pd.Interval(15, 44, closed="left"),
        pd.Interval(45, 64, closed="left"),
        pd.Interval(65, 74, closed="left"),
        pd.Interval(75, 100, closed="left"),
    ]
    index = pd.IntervalIndex(ages)
    ifr_ward = pd.DataFrame(ifr_age, index=index)
    # ***************** Imperial IFR
    ifr_age = [
        0.0,
        0.01,
        0.01,
        0.02,
        0.03,
        0.04,
        0.06,
        0.1,
        0.16,
        0.24,
        0.38,
        0.6,
        0.94,
        1.47,
        2.31,
        3.61,
        5.66,
        8.86,
        17.37,
    ]
    ages = [pd.Interval(i, i + 5, closed="left") for i in range(0, 90, 5)]
    ages += [pd.Interval(90, 100, closed="left")]
    index = pd.IntervalIndex(ages)
    ifr_imperial = pd.DataFrame(ifr_age, index=index)
    ages = np.arange(0, 100)
    male_drs = []
    female_drs = []
    care_home_male_drs = []
    care_home_female_drs = []
    all_drs = []
    hospital_male_drs = []
    hospital_female_drs = []
    hospital_all_drs = []
    for age in ages:
        male_dr = rates.get_infection_fatality_rate(age=age, sex="male")
        care_home_male_dr = rates.get_infection_fatality_rate(
            age=age, sex="male", is_care_home=True
        )

        female_dr = rates.get_infection_fatality_rate(age=age, sex="female")
        care_home_female_dr = rates.get_infection_fatality_rate(
            age=age, sex="female", is_care_home=True
        )
        male_drs.append(male_dr)
        care_home_male_drs.append(care_home_male_dr)
        female_drs.append(female_dr)
        care_home_female_drs.append(care_home_female_dr)
        male_pop = rates.population_by_age_sex_df.loc[age, "male"]
        female_pop = rates.population_by_age_sex_df.loc[age, "female"]
        all_drs.append(
            (male_pop * male_dr + female_pop * female_dr) / (male_pop + female_pop)
        )
        hospital_male_dr = rates.get_hospital_infection_fatality_rate(
            age=age, sex="male"
        )
        hospital_female_dr = rates.get_hospital_infection_fatality_rate(
            age=age, sex="female"
        )
        hospital_male_drs.append(hospital_male_dr)
        hospital_female_drs.append(hospital_female_dr)
        hospital_all_drs.append(
            (male_pop * hospital_male_dr + female_pop * hospital_female_dr)
            / (male_pop + female_pop)
        )
    # ******************* IFR comparison
    plt.bar(
        x=[index.mid for index in ifr_ward.index],
        height=ifr_ward[0].values,
        width=[index.right - index.left for index in ifr_ward.index],
        alpha=0.4,
        label="Ward",
    )
    plt.bar(
        x=[index.mid for index in ifr_imperial.index],
        height=ifr_imperial[0].values,
        width=[index.right - index.left for index in ifr_imperial.index],
        alpha=0.4,
        label="Imperial",
    )
    plt.plot(ages, 100 * np.array(male_drs), label="male IFR (Community)", color="C0")
    plt.plot(
        ages,
        100 * np.array(care_home_male_drs),
        label="male IFR (Care home)",
        color="C0",
        linestyle="--",
    )
    plt.plot(
        ages, 100 * np.array(female_drs), label="female IFR (Community)", color="C1"
    )
    plt.plot(
        ages,
        100 * np.array(care_home_female_drs),
        label="female IFR (Care Home)",
        color="C1",
        linestyle="--",
    )
    plt.plot(ages, 100 * np.array(all_drs), label="average IFR", color="C2")
    plt.plot(
        ages,
        100 * np.array(hospital_male_drs),
        label="hospital male IFR",
        linestyle="--",
        color="C0",
    )
    plt.plot(
        ages,
        100 * np.array(hospital_female_drs),
        label="hospital female IFR",
        linestyle="--",
        color="C1",
    )
    plt.plot(
        ages,
        100 * np.array(hospital_all_drs),
        label="hospital average IFR",
        linestyle="--",
        color="C2",
    )
    plt.legend()
    plt.ylabel("IFR")
    plt.xlabel("Age")
    plt.show()
