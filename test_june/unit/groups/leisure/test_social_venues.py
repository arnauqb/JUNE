import numpy as np
from june.groups.leisure import (
    SocialVenue, 
    SocialVenues, 
    group_factory, 
    supergroup_factory,
    generate_social_venues_for_world,
    generate_social_venues_for_config
)
from june.demography.geography import Geography, SuperAreas
from june.world import generate_world_from_geography
from june import paths


def test__social_venue_from_coordinates():
    super_areas = ["E02004935", "E02004940"] 
    geo = Geography.from_file({"super_area" : super_areas})
    coordinate_list = np.array([[51.752179, -0.334667], [51.741485, -0.336645]])
    social_venues = SocialVenues.from_coordinates(coordinate_list, super_areas=geo.super_areas)
    social_venues.add_to_super_areas(geo.super_areas)
    assert len(social_venues) == 2
    assert social_venues[0].super_area == geo.super_areas[0]
    assert social_venues[1].super_area == geo.super_areas[1]

def test__get_closest_venues():
    coordinate_list = np.array([[51.752179, -0.334667], [51.741485, -0.336645]])
    social_venues = SocialVenues.from_coordinates(coordinate_list, super_areas=None)
    social_venues.make_tree()
    venue = social_venues.get_closest_venues([50, 0])[0]
    assert venue == social_venues[1]

    venues_in_radius = social_venues.get_venues_in_radius([51.7, -0.33], 10)
    assert venues_in_radius[0] == social_venues[1]
    assert venues_in_radius[1] == social_venues[0]

def test__make_group_from_factory():
    KarateClub = group_factory("karate_club")
    kc = KarateClub()
    assert isinstance(kc,KarateClub)
    assert isinstance(kc,SocialVenue)
    assert KarateClub.__name__ == "KarateClub"
    assert kc.__class__.__name__ == "KarateClub"

def test__make_supergroup_from_factory():
    super_areas = ["E02004935", "E02000977"] 
    geo = Geography.from_file({"super_area" : super_areas})
    coordinate_list = np.array([[51.752179, -0.334667], [51.741485, -0.336645]])
    KarateClubs, KarateClub = supergroup_factory("karate_clubs","karate_club",return_group=True)
    assert KarateClubs.social_venue_class == KarateClub
    assert KarateClubs.__name__ == "KarateClubs"

    kcs = KarateClubs.from_coordinates(coordinate_list, super_areas=geo.super_areas)
    kcs.add_to_super_areas(geo.super_areas)

    assert isinstance(kcs[0],KarateClub)

def test__default_leisure_coordinates_filename():
    super_areas = ["E02003725", "E02003728"] 
    geo = Geography.from_file({"super_area" : super_areas})
    Cinemas, Cinema = supergroup_factory("cinemas", "cinema", return_group=True)
    default_filename = paths.data_path / paths.Path('input/leisure/cinemas_per_super_area.csv')
    assert Cinemas.default_coordinates_filename == default_filename

    cinemas = Cinemas.for_super_areas(geo.super_areas)
    assert cinemas[0].super_area in geo.super_areas
    assert cinemas[2].super_area in geo.super_areas

def test__generate_social_venues_for_world():
    geography = Geography.from_file({"super_area": ["E02002135"]})
    world = generate_world_from_geography(
        geography, include_households=False, include_commute=False
    )
    list_of_leisure_groups = ["pubs", "cinemas", "groceries"]
    list_of_singular_names = ["pub", "cinema", "grocery"]
    world.social_venues = generate_social_venues_for_world(
        list_of_leisure_groups, world, 
        list_of_singular_names=list_of_singular_names
    )

    assert len(world.social_venues["pubs"]) == 15
    assert len(world.social_venues["groceries"]) == 3
    assert len(world.social_venues["cinemas"]) == 1

def test__generate_social_venues_from_config():
    geography = Geography.from_file({"super_area": ["E02002135"]})
    world = generate_world_from_geography(
        geography, include_households=False, include_commute=False
    )
    test_social_venue_config_path = paths.configs_path / "tests" / "test_social_venue_leisure.yaml"
    world.social_venues = generate_social_venues_for_config(world,test_social_venue_config_path)

    assert len(world.social_venues["groceries"]) == 3

    # silly alt. name in config for "pubs" to make sure definitely reading coords from config.
    assert len(world.social_venues["saloons"]) == 15
    assert world.social_venues["saloons"][0].max_size == np.inf

    # pick out the correct singular names?
    assert world.social_venues["groceries"][0].spec == "grocery"
    assert world.social_venues["groceries"][0].max_size == 150

    assert set(world.social_venues.keys()) == set(["groceries", "cinemas", "saloons"])
    