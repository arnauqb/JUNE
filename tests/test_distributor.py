import numpy as np
import os
from covid.inputs import create_input_dict
from covid.classes import World, Person, Postcode, Household
from covid.distributor import populate_world


def test_global():

    census_dict = create_input_dict()
    for key, value in census_dict.items():
        census_dict[key] = census_dict[key].sample(n=5, random_state=111)
    census_dict_safe = census_dict.copy()
    world = World(census_dict)
    populate_world(world)

    n_residents_est = sum(
        [world.postcodes[i].n_residents for i in range(len(world.postcodes))]
    )
    n_residents = census_dict_safe["n_residents"].sum()

    assert n_residents == n_residents_est

    n_residents_est = sum(
        [len(world.postcodes[i].people) for i in range(len(world.postcodes))]
    )

    assert n_residents == n_residents_est

    n_households_est = sum(
        [world.postcodes[i].n_households for i in range(len(world.postcodes))]
    )
    n_households = census_dict_safe["n_households"].sum()

    assert n_households == n_households_est

    # n_households_est = sum(
    #    [len(world.postcodes[i].households) for i in range(len(world.postcodes))]
    # )
    # assert n_households == n_households_est


def test_per_postcode():

    census_dict = create_input_dict()
    for key, value in census_dict.items():
        census_dict[key] = census_dict[key].sample(n=5, random_state=111)
    census_dict_safe = census_dict.copy()

    world = World(census_dict)
    populate_world(world)

    n_residents_est = [
        world.postcodes[i].n_residents for i in range(len(world.postcodes))
    ]

    np.testing.assert_equal(n_residents_est, census_dict_safe["n_residents"].values)

    n_households_est = [
        world.postcodes[i].n_households for i in range(len(world.postcodes))
    ]

    np.testing.assert_equal(n_households_est, census_dict_safe["n_households"].values)


def compute_frequency(world, attribute):
    print(attribute)
    frequencies = []
    decoder = getattr(world, "decoder_" + attribute)
    for i in world.postcodes.keys():
        freq = np.zeros(len(decoder))
        if 'house' not in attribute:
            for j in world.postcodes[i].people.keys():
                freq[getattr(world.postcodes[i].people[j], attribute)] += 1
            freq /= world.postcodes[i].n_residents
        else:
            for j in world.postcodes[i].households.keys():
                freq[getattr(world.postcodes[i].households[j], attribute)] += 1
            freq /= world.postcodes[i].n_households

        frequencies.append(freq)
    frequencies = np.asarray(frequencies)
    assert frequencies.shape == (len(world.postcodes), len(decoder))
    return frequencies


def test_frequencies():

    census_dict = create_input_dict()
    for key, value in census_dict.items():
        census_dict[key] = census_dict[key].sample(n=5, random_state=111)
    census_dict_safe = census_dict.copy()

    world = World(census_dict)
    populate_world(world)

    for key, value in census_dict_safe.items():
        if "freq" in key:
            attribute = key.split("_")
            attribute = "_".join(attribute[:-1])
            frequencies = compute_frequency(world, attribute)
            atol_matrix = 1./np.sqrt(census_dict_safe["n_residents"]*census_dict_safe[key].values)
            for i in frequencies.shape[0]:
                for j in frequencies.shape[1]:
                    np.testing.assert_allclose(
                        frequencies[i,j],
                        census_dict_safe[key].values[i,j],
                        atol=atol_matrix[i,j]
                    )

def test_lonely_children():
    census_dict = create_input_dict()
    for key, value in census_dict.items():
        census_dict[key] = census_dict[key].sample(n=2, random_state=111)
    census_dict_safe = census_dict.copy()

    world = World(census_dict)
    populate_world(world)
    attribute = 'age'
    decoder = getattr(world, "decoder_" + attribute)
    only_children = 0
    for i in world.postcodes.keys():
        for j in world.postcodes[i].households.keys():
            freq = np.zeros(len(decoder))
            for k in world.postcodes[i].households[j].residents.keys():
                freq[getattr(world.postcodes[i].households[j].residents[k], attribute)] += 1
                # if no adults, but at least one child
                if (np.sum(freq[5:]) == 0.) & (np.sum(freq[:5]) > 0.):
                    only_children += 1

    assert only_children == 0
if __name__ == "__main__":
    test_lonely_children()
