"""Forestry districts (лесничества) with geographic bounding boxes.

Each district defines the monitoring zone assigned to rangers
who register for it. When adding a new district, just add an entry
to the DISTRICTS dict.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class District:
    slug: str
    name_ru: str
    region_ru: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


DISTRICTS: dict[str, District] = {
    "varnavino": District(
        slug="varnavino",
        name_ru="Варнавинское лесничество",
        region_ru="Нижегородская область",
        lat_min=57.05,
        lat_max=57.55,
        lon_min=44.60,
        lon_max=45.40,
    ),
    # --- Sub-districts (участковые лесничества) ---
    "mdalskoe": District(
        slug="mdalskoe",
        name_ru="Мдальское",
        region_ru="Нижегородская область",
        lat_min=57.40,
        lat_max=57.55,
        lon_min=44.60,
        lon_max=44.80,
    ),
    "semyonborskoe": District(
        slug="semyonborskoe",
        name_ru="Семёнборское",
        region_ru="Нижегородская область",
        lat_min=57.35,
        lat_max=57.50,
        lon_min=44.80,
        lon_max=45.00,
    ),
    "poplyvinskoye": District(
        slug="poplyvinskoye",
        name_ru="Поплывинское",
        region_ru="Нижегородская область",
        lat_min=57.30,
        lat_max=57.45,
        lon_min=45.00,
        lon_max=45.20,
    ),
    "kamennikoskoye": District(
        slug="kamennikoskoye",
        name_ru="Каменниковское",
        region_ru="Нижегородская область",
        lat_min=57.20,
        lat_max=57.35,
        lon_min=44.60,
        lon_max=44.80,
    ),
    "varnavinskoye": District(
        slug="varnavinskoye",
        name_ru="Варнавинское (участковое)",
        region_ru="Нижегородская область",
        lat_min=57.15,
        lat_max=57.30,
        lon_min=44.80,
        lon_max=45.00,
    ),
    "kolesnikovskoye": District(
        slug="kolesnikovskoye",
        name_ru="Колесниковское",
        region_ru="Нижегородская область",
        lat_min=57.10,
        lat_max=57.25,
        lon_min=45.00,
        lon_max=45.20,
    ),
    "kameshnoye": District(
        slug="kameshnoye",
        name_ru="Камешное",
        region_ru="Нижегородская область",
        lat_min=57.05,
        lat_max=57.20,
        lon_min=45.10,
        lon_max=45.30,
    ),
    "kayskoye": District(
        slug="kayskoye",
        name_ru="Кайское",
        region_ru="Нижегородская область",
        lat_min=57.05,
        lat_max=57.20,
        lon_min=45.20,
        lon_max=45.40,
    ),
}
