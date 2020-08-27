import pandas as pd
import json

from june.paths import data_path

default_composition_df = pd.read_csv(
    data_path
    / "not_used_in_code/census_data/households/household_composition_per_area.csv",
)
default_composition_df = default_composition_df.drop(columns=default_composition_df.columns[[0, 1, 3, 4]])
default_composition_df = default_composition_df.set_index("geography")

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
    default_composition_df=default_composition_df,
    students_df=default_students_df,
    communal_df=default_communal_df,
    output_area="E00081788",
):
    data = default_composition_df.loc[output_area]
    students = students_df.loc[output_area]
    communal = communal_df.loc[output_area]
    ret = {}

    # student
    ret["student"] = {}
    ret["student"][0] = {}
    ret["student"][0]["residents"] = int(students.values[0])
    ret["student"][0]["number"] = int(data[
        "Household Composition: Other household types: All full-time students; measures: Value"
    ])

    # single
    ret["single"] = {}
    ret["single"][0] = {}
    ret["single"][0]["type"] = "old"
    ret["single"][0]["number"] = int(data[
        "Household Composition: One person household: Aged 65 and over; measures: Value"
    ])
    ret["single"][1] = {}
    ret["single"][1]["type"] = "adult"
    ret["single"][1]["number"] = int(data[
        "Household Composition: One person household: Other; measures: Value"
    ])

    # couple
    ret["couple"] = {}
    ret["couple"][0] = {}
    ret["couple"][0]["type"] = "old"
    ret["couple"][0]["number"] = int(data[
        "Household Composition: One family only: All aged 65 and over; measures: Value"
    ])
    ret["couple"][1] = {}
    ret["couple"][1]["type"] = "adult"
    ret["couple"][1]["number"] = int(
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
    ret["family"][0]["number"] = int(data[
        "Household Composition: One family only: Lone parent: One dependent child; measures: Value"
    ])

    ret["family"][1] = {}
    ret["family"][1]["adults"] = 1
    ret["family"][1]["kids"] = ">=2"
    ret["family"][1]["young_adults"] = ">=0"
    ret["family"][1]["number"] = int(data[
        "Household Composition: One family only: Lone parent: Two or more dependent children; measures: Value"
    ])

    ret["family"][2] = {}
    ret["family"][2]["adults"] = 1
    ret["family"][2]["kids"] = 0
    ret["family"][2]["young_adults"] = ">=1"
    ret["family"][2]["number"] = int(data[
        "Household Composition: One family only: Lone parent: All children non-dependent; measures: Value"
    ])

    ret["family"][3] = {}
    ret["family"][3]["adults"] = 2
    ret["family"][3]["kids"] = 1
    ret["family"][3]["young_adults"] = ">=0"
    ret["family"][3]["number"] = int(
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
    ret["family"][4]["number"] = int(
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
    ret["family"][5]["number"] = int(
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
    ret["other"][0]["number"] = int(data[
        "Household Composition: Other household types: With one dependent child; measures: Value"
    ])

    ret["other"][1] = {}
    ret["other"][1]["adults"] = ">=1"
    ret["other"][1]["kids"] = ">=2"
    ret["other"][1]["young_adults"] = ">=0"
    ret["other"][1]["old_adults"] = ">=0"
    ret["other"][1]["number"] = int(data[
        "Household Composition: Other household types: With two or more dependent children; measures: Value"
    ])

    ret["other"][2] = {}
    ret["other"][2]["adults"] = 0
    ret["other"][2]["kids"] = 0
    ret["other"][2]["young_adults"] = 0
    ret["other"][2]["old_adults"] = ">=1"
    ret["other"][2]["number"] = int(data[
        "Household Composition: Other household types: All aged 65 and over; measures: Value"
    ])

    ret["other"][3] = {}
    ret["other"][3]["adults"] = ">=0"
    ret["other"][3]["kids"] = ">=0"
    ret["other"][3]["young_adults"] = ">=0"
    ret["other"][3]["old_adults"] = ">=0"
    ret["other"][3]["number"] = int(data[
        "Household Composition: Other household types: Other; measures: Value"
    ])

    # communal
    ret["communal"] = {}
    ret["communal"][0] = {}
    ret["communal"][0]["number"] = int(communal["establishments"])
    ret["communal"][0]["residents"] = int(communal["residents"])
    return ret

if __name__ == "__main__":
    from tqdm import tqdm
    total_dict = {}
    pbar = tqdm(total = len(default_composition_df))
    for area in default_composition_df.index:
        total_dict[area] = read_household_comp(output_area=area)
        pbar.update(1)
    with open("test2.json", "w") as f:
        json.dump(total_dict, f)


