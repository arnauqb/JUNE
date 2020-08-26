import pandas as pd

from june.paths import data_path

default_sizes_df = pd.read_csv(
    data_path / "not_used_in_code/census_data/households/household_size_per_area.csv",
    index_col=0,
)
default_compositions_df = pd.read_csv(
    data_path
    / "not_used_in_code/census_data/households/household_composition_per_area.csv",
    index_col=0,
)
default_communal_df = pd.read_csv(
    data_path
    / "not_used_in_code/census_data/households/communal_residents_and_establishments_per_area.csv",
    index_col=0,
)
default_students_df = pd.read_csv(
    data_path / "not_used_in_code/census_data/households/n_students_per_area.csv",
    index_col=0,
)


def read_household_comp(
    comp_raw_df=default_compositions_df,
    students_df=default_students_df,
    communal_df=default_communal_df,
    output_area="E00081788",
):
    data = comp_raw_df.loc[output_area]
    students = students_df.loc[output_area]
    communal = communal_df.loc[output_area]
    ret = {}

    # student
    ret["student"] = {}
    ret["student"][0] = {}
    ret["student"][0]["total_students"] = students.values[0]
    ret["student"][0]["number"] = data[
        "Household Composition: Other household types: All full-time students; measures: Value"
    ]

    # single
    ret["single"] = {}
    ret["single"][0] = {}
    ret["single"][0]["type"] = "old"
    ret["single"][0]["number"] = data[
        "Household Composition: One person household: Aged 65 and over; measures: Value"
    ]
    ret["single"][1] = {}
    ret["single"][1]["type"] = "adult"
    ret["single"][1]["number"] = data[
        "Household Composition: One person household: Other; measures: Value"
    ]

    # couple
    ret["couple"] = {}
    ret["couple"][0] = {}
    ret["couple"][0]["type"] = "old"
    ret["couple"][0]["number"] = data[
        "Household Composition: One family only: All aged 65 and over; measures: Value"
    ]
    ret["couple"][1] = {}
    ret["couple"][1]["type"] = "adult"
    ret["couple"][1]["number"] = (
        data[
            "Household Composition: One family only: Married couple: No children; measures: Value"
        ]
        + data[
            "Household Composition: One family only: Cohabiting couple: No children; measures: Value"
        ]
        + data[
            "Household Composition: One family only: Same-sex civil partnership couple: Total; measures: Value"
        ]
    )

    # family
    ret["family"] = {}
    ret["family"][0] = {}
    ret["family"][0]["adults"] = 1
    ret["family"][0]["kids"] = 1
    ret["family"][0]["young_adults"] = ">=0"
    ret["family"][0]["number"] = data[
        "Household Composition: One family only: Lone parent: One dependent child; measures: Value"
    ]

    ret["family"][1] = {}
    ret["family"][1]["adults"] = 1
    ret["family"][1]["kids"] = ">=2"
    ret["family"][1]["young_adults"] = ">=0"
    ret["family"][1]["number"] = data[
        "Household Composition: One family only: Lone parent: Two or more dependent children; measures: Value"
    ]

    ret["family"][2] = {}
    ret["family"][2]["adults"] = 1
    ret["family"][2]["kids"] = 0
    ret["family"][2]["young_adults"] = ">=1"
    ret["family"][2]["number"] = data[
        "Household Composition: One family only: Lone parent: All children non-dependent; measures: Value"
    ]

    ret["family"][3] = {}
    ret["family"][3]["adults"] = 2
    ret["family"][3]["kids"] = 1
    ret["family"][3]["young_adults"] = ">=0"
    ret["family"][3]["number"] = (
        data[
            "Household Composition: One family only: Married couple: One dependent child; measures: Value"
        ]
        + data[
            "Household Composition: One family only: Cohabiting couple: One dependent child; measures: Value"
        ]
    )

    ret["family"][4] = {}
    ret["family"][4]["adults"] = 2
    ret["family"][4]["kids"] = ">=2"
    ret["family"][4]["young_adults"] = ">=0"
    ret["family"][4]["number"] = (
        data[
            "Household Composition: One family only: Married couple: Two or more dependent children; measures: Value"
        ]
        + data[
            "Household Composition: One family only: Cohabiting couple: Two or more dependent children; measures: Value"
        ]
    )

    ret["family"][5] = {}
    ret["family"][5]["adults"] = 2
    ret["family"][5]["kids"] = 0
    ret["family"][5]["young_adults"] = ">=1"
    ret["family"][5]["number"] = (
        data[
            "Household Composition: One family only: Married couple: All children non-dependent; measures: Value"
        ]
        + data[
            "Household Composition: One family only: Cohabiting couple: All children non-dependent; measures: Value"
        ]
    )

    # other
    ret["other"] = {}
    ret["other"][0] = {}
    ret["other"][0]["adults"] = ">=1"
    ret["other"][0]["kids"] = ">=1"
    ret["other"][0]["young_adults"] = ">=0"
    ret["other"][0]["old_adults"] = ">=0"
    ret["other"][0]["number"] = data[
        "Household Composition: Other household types: With one dependent child; measures: Value"
    ]

    ret["other"][1] = {}
    ret["other"][1]["adults"] = ">=1"
    ret["other"][1]["kids"] = ">=2"
    ret["other"][1]["young_adults"] = ">=0"
    ret["other"][1]["old_adults"] = ">=0"
    ret["other"][1]["number"] = data[
        "Household Composition: Other household types: With two or more dependent children; measures: Value"
    ]

    ret["other"][2] = {}
    ret["other"][2]["adults"] = 0
    ret["other"][2]["kids"] = 0
    ret["other"][2]["young_adults"] = 0
    ret["other"][2]["old_adults"] = ">=1"
    ret["other"][2]["number"] = data[
        "Household Composition: Other household types: All aged 65 and over; measures: Value"
    ]

    ret["other"][3] = {}
    ret["other"][3]["adults"] = ">=0"
    ret["other"][3]["kids"] = ">=0"
    ret["other"][3]["young_adults"] = ">=0"
    ret["other"][3]["old_adults"] = ">=0"
    ret["other"][3]["number"] = data[
        "Household Composition: Other household types: Other; measures: Value"
    ]

    # communal
    ret["communal"] = {}
    ret["communal"][0] = {}
    ret["communal"][0]["number"] = communal["establishments"]
    ret["communal"][0]["residents"] = communal["residents"]
    return ret
