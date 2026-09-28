# SolarDSS

SolarDSS is a **standalone** Python library for deterministic solar PV siting
and screening. It performs decision algebra, constraint screening, published
terrain derivatives, and provenance recording. It does **not** estimate solar
irradiation, weather, or PV performance from implicit models.

The library has no dependency on Ariadne Thread, LLM providers, RAG, agent
orchestration, or frontend frameworks. The intended future adapter direction is:

```
Ariadne Thread
        ↓
AriadneThread-Solar
        ↓
SolarDSS
```

SolarDSS must remain usable on its own.

## Indicator registry

The scientific vocabulary lives in ``solardss.indicators.registry``. It describes
indicators, constraints, design variables, and objectives. It does **not**
download climate layers, OSM features, or grid-operator data.

```python
from solardss.indicators import DecisionMode, registry

ghi = registry.get("SOLAR_GHI_ANNUAL")
location = registry.resolve("PV_LOCATION", mode=DecisionMode.SITE_SELECTION)
```

``PV_LOCATION`` is registered once. Site selection resolves it to a controllable
decision variable; fixed-site mode resolves it to fixed context.

## Indicator engine

``IndicatorEngine`` turns a geometry (fixed site, one candidate, or a candidate
table) plus an indicator list into a structured ``IndicatorSet``. It reads the
registry, expands an **explicit** dependency DAG, selects a registered provider
per indicator, fetches, runs registered derivations, checks units, and attaches
provenance. Providers never call each other.

Dependencies are declared, not discovered. ``TERRAIN_SLOPE_MEAN`` depends on a
DEM source (a terrain provider may compute Horn slope from its own raster);
``INFRA_DISTANCE_TO_ROAD`` depends on road geometry; ``SOLAR_POA_IRRADIATION``
depends on solar resource plus tilt and azimuth. Cycles raise
``CircularDependencyError``.

Each indicator is ``available``, ``missing``, ``unsupported``,
``provider_unavailable``, or ``externally_required``. Missing values stay
missing: the engine never substitutes zero. Candidate batches prefetch a shared
AOI window once when a provider implements ``prefetch``, and reuse one fetch for
``shared_aoi`` indicators.

```python
from datetime import datetime, timezone

from solardss.data.geo import GeographicRef
from solardss.engine import IndicatorEngine
from solardss.providers import FakeSeed, FakeSolarResourceProvider, FixedClock

berlin = GeographicRef(name="Berlin", longitude=13.4, latitude=52.52)
engine = IndicatorEngine(
    [
        FakeSolarResourceProvider(
            seeds={"SOLAR_GHI_ANNUAL": FakeSeed(value=1085.0, unit="kWh/m2/year")}
        )
    ],
    clock=FixedClock(datetime(2024, 6, 1, tzinfo=timezone.utc)),
)
bundle = engine.compute(geometry=berlin, indicators=["SOLAR_GHI_ANNUAL"])
```

## Decision problems

``SolarDecisionProblem`` distinguishes **site selection** (location is chosen)
from **fixed-site** evaluation (location is given). Site-selection scopes are
region search, AOI search, and candidate ranking.

Incomplete spatial inputs are reported by ``get_missing_requirements`` as
structured ``{field, required, reason}`` records — not as dialogue. Candidate
generation and site ranking are not authorized on a fixed-site problem.

## Candidate generation

``generate_candidates`` materialises polygons for the three site-selection
scopes. It does **not** rank sites.

- **region_search** / **aoi_search** — tessellate the supplied boundary with a
  ``regular_grid`` or ``hexagonal_grid``. ``cell_size`` and ``cell_size_unit``
  are required; SolarDSS never picks a default resolution. Projected CRS use
  metres; geographic CRS use degrees. Cells are clipped to the boundary.
  Ground-mounted cells keep ``grid_row`` / ``grid_col`` so contiguous suitable
  cells can be grouped later (``contiguous_groups``). Geometries are not
  dissolved and scores are not applied.
- **candidate_ranking** — copy the caller’s features in order (parcels,
  buildings, predefined sites). No new locations are created. Rooftop
  footprints are valid candidate geometries via ``candidates_from_buildings``.

```python
from solardss.candidates import (
    CandidateGenerationConfig,
    GenerationStrategy,
    generate_candidates,
)
from solardss.data.geo import BoundingBox
from solardss.indicators import SiteSelectionScope

bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=30.0, max_lat=30.0, crs="EPSG:32632")
bundle = generate_candidates(
    scope=SiteSelectionScope.REGION_SEARCH,
    config=CandidateGenerationConfig(
        strategy=GenerationStrategy.REGULAR_GRID,
        crs="EPSG:32632",
        cell_size=10.0,
        cell_size_unit="m",
        origin_x=0.0,
        origin_y=0.0,
    ),
    boundary=bbox,
)
```

## Constraint engine

Hard and soft screening is an explicit ``ConstraintProfile`` of configuration
objects (``MaximumConstraint``, ``MinimumConstraint``,
``BooleanExclusionConstraint``, ``AllowedCategoryConstraint``,
``DisallowedCategoryConstraint``, ``OverlapConstraint``,
``DistanceConstraint``). Every numeric bound is a ``Threshold`` with
``value``, ``unit``, ``source``, ``profile``, and ``rationale``. None of those
fields is optional, and none is a universal scientific cutoff.

``UTILITY_SCALE_DEFAULT_EXAMPLE`` and ``ROOFTOP_DEFAULT_EXAMPLE`` are labelled
example profiles (``example=True``). SolarDSS never selects them from project
type. Pass a profile into ``ConstraintEngine.evaluate``.

A missing hard-constraint indicator is **not** treated as zero: the candidate
status is ``unknown`` and ``feasible`` is false. Soft hits (penalty/warning)
populate ``failed_constraints`` and ``warnings`` without excluding the site.
Boolean raster OR screening (``apply_exclusions``) is unchanged.

```python
from solardss.constraints import (
    ConstraintEffect,
    ConstraintEngine,
    ConstraintProfile,
    MaximumConstraint,
    Threshold,
)
from solardss.core.quantity import Quantity

profile = ConstraintProfile(
    profile_id="project_water",
    label="Project-specific water screen",
    constraints=(
        MaximumConstraint(
            constraint_id="max_water",
            indicator_id="LAND_WATER_FRACTION",
            effect=ConstraintEffect.HARD_EXCLUSION,
            source="project-method-note",
            profile="project_water",
            rationale="Exclude cells whose mapped water fraction exceeds the project bound.",
            threshold=Threshold(
                value=0.1,
                unit="1",
                source="project-method-note",
                profile="project_water",
                rationale="Exclude cells whose mapped water fraction exceeds the project bound.",
            ),
        ),
    ),
)
evaluation = ConstraintEngine().evaluate(
    "cell-1",
    {"LAND_WATER_FRACTION": Quantity(value=0.25, unit="1")},
    profile,
)
```

## Suitability scoring

Site ranking uses a transparent **Weighted Sum Model**. Weights are never
inferred and never produced by an LLM. Every ``CriterionWeight`` carries
``weight``, ``unit``, ``source``, ``profile``, ``rationale``, and an explicit
normalization kind (benefit, cost, or target-distance).

``ENERGY_FOCUSED``, ``INFRASTRUCTURE_FOCUSED``, ``ENVIRONMENT_FOCUSED``, and
``BALANCED`` are labelled example profiles (``example=True``). They are
configuration, not universal truth, and are not selected automatically.

Equal min/max (or zero target deviation) scores ``1.0``. Missing indicators are
not treated as zero; those candidates are omitted as incomplete. Excluded
candidates are omitted from the ranked list. ``RankingResult`` records, for
every remaining site, raw value, normalized value, weight, weighted
contribution, score, rank, and constraint status.

Multi-objective analysis is separate: ``pareto_candidates`` identifies
non-dominated sites from **raw** objective values and does not collapse them
into a WSM score.

```python
from solardss.core.status import FeasibilityStatus
from solardss.suitability import ENERGY_FOCUSED, rank_candidates

ranking = rank_candidates(
    (
        ("a", {"SOLAR_GHI_ANNUAL": 2000.0, "TERRAIN_SLOPE_MEAN": 5.0,
               "INFRA_DISTANCE_TO_POWER_LINE": 800.0, "LAND_AVAILABLE_AREA": 9000.0}),
        ("b", {"SOLAR_GHI_ANNUAL": 1500.0, "TERRAIN_SLOPE_MEAN": 4.0,
               "INFRA_DISTANCE_TO_POWER_LINE": 100.0, "LAND_AVAILABLE_AREA": 8000.0}),
    ),
    ENERGY_FOCUSED,
    constraint_status={"a": FeasibilityStatus.FEASIBLE, "b": FeasibilityStatus.FEASIBLE},
)
```

## PV component catalog

``PVModule``, ``PVInverter``, and ``BatterySystem`` are typed datasheet rows.
SolarDSS does **not** ship manufacturer products. ``empty_pv_catalog()`` is the
default. Load a caller-supplied local JSON or CSV file with ``load_pv_catalog``;
every row must include provenance, and missing electrical fields stay missing.

Rows labelled ``fixture=True`` / ``source_kind=test_fixture`` are synthetic
**TEST** products for unit tests. They are omitted from ``list_modules`` (and
the inverter/battery listings) unless ``include_fixtures=True``. They are not a
default catalog.

``ModuleFilter`` selects by technology class, efficiency, rated power,
temperature coefficient of Pmax, and dimensional limits (length, width,
thickness, area, weight). A missing datasheet field fails a bound rather than
passing. Trusted external databases (for example CEC) are reserved as
unimplemented adapters; SolarDSS does not download or invent those rows.

```python
from solardss.pv import ModuleFilter, empty_pv_catalog, load_pv_catalog

catalog = empty_pv_catalog()
assert catalog.list_modules() == ()

imported = load_pv_catalog("catalog.json")
mono = imported.list_modules(ModuleFilter(technology="mono-c-Si", min_efficiency=0.20))
```

## PV configuration generation

After the site is resolved, ``PVDesignEngine`` enumerates and evaluates
``PVSystemConfiguration`` records. Location is **not** a design variable here:
it is controllable only during site selection, and a fixed-site problem already
has it.

Evaluated variables are module, module count, installed capacity, tilt, azimuth,
mounting type, tracking mode, inverter, DC/AC ratio, row spacing, ground
coverage ratio, and optional battery size. Weights and layouts are never
produced by an LLM.

On a rooftop, a supplied ``roof_tilt_deg`` / ``roof_azimuth_deg`` is a
constraint, not a free search axis. Ground-mount tilt, azimuth, and row spacing
are decision variables. The first generator is bounded discrete enumeration
(``BoundedEnumerationGenerator``). A future solver can implement
``PVConfigurationGenerator``; ``UnimplementedOptimizer`` exists so that
interface is explicit. There is no MIP/NLP dependency.

Stated constraints are available area, roof area, requested capacity, electrical
and inverter limits, user preferences, and module dimensions. Missing datasheet
fields fail a bound or are reported ``unknown``; they are not treated as zero.
Use a caller-supplied catalog; test fixtures are omitted unless
``include_fixtures=True``.

```python
from solardss.data.geo import GeographicRef
from solardss.indicators import ProjectType
from solardss.pv import PVDesignEngine, PVDesignProblem
from solardss.pv.design import ConfigurationSearchSpace

site = GeographicRef(name="Site", longitude=13.4, latitude=52.52)
problem = PVDesignProblem(
    location=site,
    project_type=ProjectType.RESIDENTIAL_ROOFTOP,
    roof_area_m2=60.0,
    roof_tilt_deg=20.0,
    roof_azimuth_deg=180.0,
    target_capacity_w=10_000.0,
    search_space=ConfigurationSearchSpace(
        module_counts=(25,),
        tilt_deg=(20.0,),
        azimuth_deg=(180.0,),
    ),
)
engine = PVDesignEngine(catalog)
evaluated = engine.generate(problem)
```

## Annual PV energy yield (PVGIS)

``PVGISYieldEngine`` estimates annual AC energy with official PVGIS 5.3
``PVcalc``. It is **not** the solar-resource provider
(``PVGISSolarResourceProvider`` / ``MRcalc``) and it does not use
``energy_from_capacity_factor``. Annual generation is never computed by an LLM.

The caller supplies location, installed capacity, tilt, azimuth, mounting,
tracking, and system loss. Module technology is sent only when PVGIS supports
it (``crystSi``, ``CIS``, ``CdTe``). SolarDSS azimuth (clockwise from north) is
converted to PVGIS ``aspect`` (0 = south). Rooftop maps to
``mountingplace=building``, ground-mount to ``free``. System loss is obligatory;
SolarDSS does not default 14%.

Every numeric field is tagged ``pvgis_provider`` (PVGIS-published) or
``solardss_derived`` (an identity such as specific yield ``E_y / peakpower``).
Missing PVGIS fields stay missing. Requests are cached on ``PVGISClient``.
Unit tests use the stored ``PVcalc`` fixture; live calls are opt-in via
``SOLARDSS_PVGIS_LIVE=1``.

```python
from solardss.data.geo import GeographicRef
from solardss.pv import PVGISYieldEngine, PVYieldRequest
from solardss.pv.design import MountingType, TrackingMode

request = PVYieldRequest(
    location=GeographicRef(longitude=8.0, latitude=45.0),
    installed_capacity_kw=1.0,
    tilt_deg=35.0,
    azimuth_deg=180.0,
    mounting=MountingType.GROUND,
    tracking=TrackingMode.FIXED,
    technology="mono-c-Si",
    system_loss_percent=14.0,
)
engine = PVGISYieldEngine()  # tests inject cache + MockTransport; live is opt-in
result = engine.evaluate(request)
```

## Detailed PV energy yield (pvlib, optional)

``PvlibYieldEngine`` implements the same ``PVYieldEngine`` protocol as
``PVGISYieldEngine``. It runs an explicit physics chain with official pvlib
APIs: weather → solar position → POA transposition → module temperature →
DC → inverter → AC → losses → hourly / monthly / annual energy.

pvlib is an extra (``pip install 'solardss[pvlib]'``). CEC/Sandia databases are
not loaded. Missing DNI/DHI is not filled by a decomposition model. Temperature
coefficients, inverter efficiency, albedo, and tracker geometry are not
inferred from mounting type. Documented defaults (for example NREL PVWatts
``eta_inv_ref`` and STC ``temp_ref``) are recorded on the result.

```python
from datetime import datetime, timezone

from solardss.data.geo import GeographicRef
from solardss.pv import PvlibYieldEngine, PVYieldRequest, WeatherObservation
from solardss.pv.design import MountingType, TrackingMode

request = PVYieldRequest(
    location=GeographicRef(longitude=0.0, latitude=0.0),
    elevation_m=1400.0,
    timezone="UTC",
    installed_capacity_kw=1.0,
    tilt_deg=0.0,
    azimuth_deg=180.0,
    mounting=MountingType.GROUND,
    tracking=TrackingMode.FIXED,
    system_loss_percent=0.0,
    gamma_pdc=-0.003,
    inverter_efficiency=0.96,
    temperature_model_parameters={"a": -3.47, "b": -0.0594, "deltaT": 3.0},
    weather=(
        WeatherObservation(
            timestamp=datetime(2020, 3, 20, 12, 0, tzinfo=timezone.utc),
            ghi_w_m2=1000.0,
            dni_w_m2=850.0,
            dhi_w_m2=150.0,
            temp_air_c=25.0,
            wind_speed_m_s=2.0,
        ),
    ),
    interval_hours=1.0,
    weather_source="caller_supplied",
)
result = PvlibYieldEngine().evaluate(request)
```

## Data providers and provenance

Every retrieved indicator value is an ``IndicatorObservation`` with
``DataProvenance``, ``SpatialCoverage``, and ``TemporalCoverage``. The
observation records source, retrieval time, geographic and temporal resolution,
dataset version when known, raw vs derived origin, calculation method, quality
flags, and assumptions.

Providers implement ``availability``, ``coverage``, ``fetch``, and ``health``.
Missing data is a structured status (unavailable, unsupported region, missing
coverage, temporary failure, not configured) — never a fabricated number.

This package ships in-memory **fake** providers, a deterministic local cache, an
official **PVGIS 5.3** HTTP adapter (``solardss.data.pvgis``), an optional
**ERA5-Land** weather adapter (``solardss.data.era5_land``), a Copernicus-
compatible **DEM terrain** adapter (``solardss.data.dem``), an ESA
**WorldCover** land-cover adapter (``solardss.data.worldcover``), a bounded
**OpenStreetMap / Overpass** adapter (``solardss.data.osm``), a pluggable
**protected-area** adapter (``solardss.data.protected``), and a pluggable
**hazard** adapter (``solardss.data.hazard``). Cache keys are
SHA-256 hashes of the provider id and request. There is no default home-directory
cache path: pass a directory to ``DirectoryCache`` or set ``SOLARDSS_CACHE_DIR``.

PVGIS field names and units are taken from the public JSON contract of ``MRcalc``
and ``PVcalc``. Missing PVGIS fields stay missing. ``MRcalc`` feeds the solar-
resource provider; annual PV energy yield is a separate ``PVcalc`` engine
(``PVGISYieldEngine``). Unit tests use recorded fixtures; live calls are opt-in
via ``SOLARDSS_PVGIS_LIVE=1``.

ERA5-Land is optional because Copernicus CDS access needs a personal token.
Configure ``SOLARDSS_ERA5_LAND_PATH`` for a local NetCDF/Zarr subset, or
``CDSAPI_KEY`` / ``SOLARDSS_CDSAPI_KEY`` for CDS. Credentials are never stored
in source or provenance. Missing credentials yield ``not_configured``, not a
crash. Install the extra with ``pip install 'solardss[era5]'``. Cloud cover is
not an ERA5-Land variable and is not mapped unless the opened file actually
contains it. Live CDS calls are opt-in via ``SOLARDSS_ERA5_LIVE=1``.

DEM terrain analysis (Horn slope/aspect, Riley TRI, zonal elevation) is
computed from a caller-supplied raster or an injected retriever. There is no
built-in Copernicus STAC/OpenSearch downloader: pass a ``DEMRasterSource``,
``SOLARDSS_DEM_PATH``, or a retriever callable. Geographic Copernicus GLO-30
windows are reprojected to a local UTM CRS before slope is computed. Horizon
obstruction and terrain shading are reserved and return ``unavailable``.
Unit tests use in-memory synthetic DEMs with analytically known planes.

Land-cover fractions and dominant class are computed from a classified raster
plus an explicit legend. ESA WorldCover 10 m (v200 / 2021) is the first dataset;
the provider interface accepts any north-up classified raster. Available land
area requires an ``AvailableLandProfile`` of available and excluded classes —
non-water pixels are never treated as PV-suitable automatically. Configure
``SOLARDSS_WORLDCOVER_PATH`` or inject a ``LandCoverRasterSource``. Classified
windows are not bilinearly resampled. Unit tests use small rasters with known
class proportions.

**OpenStreetMap** (``solardss.data.osm``) is a bounded Overpass adapter. It
never issues a global or unfiltered query: every request has a geographic bbox
capped at 1° / 50 km. Distances to power lines, substations, roads, and
settlements are straight-line (projected or authalic-equirectangular), not
routing distance. Grid hosting capacity is **not** derived from OSM. Building
footprint area is plan-view geometry only. ``roof:shape``, ``roof:direction``,
``roof:angle``, and ``building:levels`` are preserved when present;
``ROOF_ORIENTATION`` / ``ROOF_SLOPE`` are mapped only from those tags.
``INFRA_DISTANCE_TO_DEMAND_CENTER`` requires caller-declared coordinates.
Provenance ``source_name`` is ``OpenStreetMap`` and includes
``timestamp_osm_base`` when Overpass returns it. Tests mock HTTP and use
fixed extracts.

**Protected areas** (``solardss.data.protected``) are region-specific. There is
no global unrestricted cadastre and no WDPA download. The generic vector
provider reads GeoJSON without GIS extras, and GeoPackage or shapefile when
``geopandas`` is installed. Official national datasets and European products
(Natura 2000, CDDA, licensed WDPA extracts) are injected as regional adapters
with an explicit field map. ``LAND_PROTECTED_AREA_OVERLAP`` is the intersection
fraction plus area, percent, site identifiers, protection categories, and
source provenance. Hard exclusion, penalty, or warning come only from a
``ProtectedAreaConstraintProfile`` — overlap is never a universal exclusion.
Configure ``SOLARDSS_PROTECTED_AREA_PATH`` or inject a ``ProtectedAreaSource``.
The named opt-in profile ``iucn_ia_ib_hard_exclusion`` excludes IUCN Ia/Ib
only.

**Hazards** (``solardss.data.hazard``) are sampled from a caller-supplied
raster or vector. There is no global flood, wildfire, landslide, or hail
dataset and no default Low/Medium/High scale. Generic raster/vector adapters
come first; ``FloodRiskProvider``, ``WildfireRiskProvider``,
``LandslideRiskProvider``, and ``HailRiskProvider`` bind one kind to
``HAZARD_FLOOD_RISK``, ``HAZARD_WILDFIRE_RISK``, ``HAZARD_LANDSLIDE_RISK``,
or ``HAZARD_HAIL_RISK``. Categories are applied only from an explicit
``HazardValueMapping`` for that source. Each sample records raw value,
optional mapped category/score, source, resolution, return period, year/version,
and uncertainty/quality metadata. Configure ``SOLARDSS_HAZARD_FLOOD_PATH``,
``SOLARDSS_HAZARD_WILDFIRE_PATH``, ``SOLARDSS_HAZARD_LANDSLIDE_PATH``, or
``SOLARDSS_HAZARD_HAIL_PATH`` — there is no ``SOLARDSS_HAZARD_PATH``.
JSON grids and GeoJSON work without GIS extras. Use
``hazard_available()``, ``hazard_for_site()``, and ``hazard_for_candidates()``.

```python
from datetime import datetime, timezone

from solardss.data.geo import GeographicRef
from solardss.providers import (
    FakeSeed,
    FakeSolarResourceProvider,
    FetchRequest,
    FixedClock,
    MemoryCache,
)

berlin = GeographicRef(name="Berlin", longitude=13.4, latitude=52.52)
provider = FakeSolarResourceProvider(
    seeds={"SOLAR_GHI_ANNUAL": FakeSeed(value=1085.0, unit="kWh/m2/year")},
    cache=MemoryCache(),
    clock=FixedClock(datetime(2021, 6, 1, tzinfo=timezone.utc)),
)
observation = provider.fetch(
    FetchRequest(indicator_id="SOLAR_GHI_ANNUAL", location=berlin)
)
```

## What this package computes

| Method | Module | Meaning |
| --- | --- | --- |
| Min-max normalization | `solardss.indicators` | Benefit/cost rescaling onto `[0, 1]` with an explicit or sample range |
| Weighted linear combination | `solardss.suitability` | `sum(w_i * x_i)` with caller-supplied weights |
| Weighted Sum Model ranking | `solardss.suitability` | Explicit decision profile; per-criterion trace; excluded omitted |
| Pareto non-dominated set | `solardss.suitability` | Raw maximize/minimize objectives; no scalar score |
| AHP geometric-mean weights | `solardss.suitability` | Priority vector and Saaty consistency ratio from a pairwise matrix |
| Boolean exclusion | `solardss.constraints` | A cell/record is excluded if any exclusion layer is true |
| Constraint engine | `solardss.constraints` | Hard/soft tests from an explicit profile; missing values stay missing |
| Candidate extraction | `solardss.candidates` | Feasible mask cells become a candidate table |
| Spatial candidates | `solardss.candidates` | Region/AOI grids (configured cell size) or supplied features; no ranking |
| Greedy budget selection | `solardss.optimization` | Score-descending selection until a cost budget is exhausted |
| Horn (1981) slope/aspect | `solardss.terrain` | 8-neighbor derivatives on a projected elevation grid |
| Riley (1999) TRI | `solardss.terrain` | 8-neighbor terrain ruggedness index in metres |
| DEM zonal terrain | `solardss.data.dem` | Elevation/slope/aspect/TRI statistics from a windowed DEM |
| Land-cover zonal fractions | `solardss.indicators.land` | Area, percent, and dominant class from a classified map |
| Available land area | `solardss.indicators.land` | Profile of available/excluded classes, not “all non-water” |
| OSM nearest-feature distance | `solardss.indicators.infrastructure` | Straight-line distance in a bounded OSM window |
| OSM building footprint | `solardss.indicators.rooftop` | Plan-view area; roof tags mapped only when present |
| Protected-area overlap | `solardss.indicators.protected` | Cadastre intersection; disposition from a constraint profile |
| Hazard source mapping | `solardss.indicators.hazard` | Sample a supplied map; categories only from that dataset's legend |
| Indicator engine | `solardss.engine` | DAG-ordered fetch and derivation into an `IndicatorSet` |
| Capacity-factor energy | `solardss.pv` | `E = P_rated * CF * t` when **all** terms are supplied |
| Irradiation-area-efficiency energy | `solardss.pv` | `E = G * A * eta` when **all** terms are supplied |
| PV catalog load | `solardss.pv` | Validate caller-supplied module/inverter/battery rows; default catalog is empty |
| PV design evaluate | `solardss.pv` | Screen one configuration against area, capacity, electrical, and preference bounds |
| PV design enumerate | `solardss.pv` | Bounded discrete configuration search after the site is resolved; not a global optimizer |

## What this package does not compute

- A default candidate-grid resolution. Region and AOI search require an explicit
  ``cell_size`` / ``cell_size_unit``; coordinates are not reprojected
- Solar irradiation (GHI/DNI/DHI), clear-sky models, or typical meteorological years
- PV module, inverter, shading, soiling, or temperature derating models
- A manufacturer module database. The default catalog is empty; import a local
  CSV/JSON file or plug in a future trusted adapter. Test fixtures are not
  commercial products
- Global spatial optimization or MILP site selection
- A globally optimal PV layout. Configuration search is bounded enumeration of
  an explicit discrete space; rooftop tilt/azimuth are constraints when the roof
  plane is given
- Horizon obstruction angles or terrain shading from a DEM
- Grid hosting capacity from OpenStreetMap or from the presence of a line
- Usable roof area, roof shading, or structural capacity from a footprint
- A worldwide unrestricted protected-area layer, or automatic exclusion of every
  protection class. GeoPackage/shapefile I/O needs the optional GIS extra;
  GeoJSON overlap does not. GIS extras are also used to read GeoTIFF DEMs,
  classified land-cover rasters, or hazard GeoTIFFs/GeoPackages and to
  reproject geographic windows; they are not required to import the library
  or to run Horn/TRI, land-cover zonal statistics, protected-area overlap on
  in-memory rings, or hazard sampling of JSON grids and GeoJSON
- Flood, wildfire, landslide, or hail risk models, or Low/Medium/High classes
  inferred from raster statistics. Water proximity is not flood risk; land
  cover is not wildfire risk; slope is not landslide risk

Missing physical inputs are errors or explicit `incomplete_inputs` status.
Empty results stay empty. Nothing is invented.

## Install

Python 3.10 or newer. Use a virtual environment and pip (not uv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional GIS stack (GeoTIFF DEM, land-cover, and hazard windowing, CRS
transforms, GeoPackage and shapefile protected-area or hazard files):

```bash
pip install -e ".[gis]"
```

Optional ERA5-Land / Copernicus CDS extra:

```bash
pip install -e ".[era5]"
```

Optional pvlib detailed PV yield engine:

```bash
pip install -e ".[pvlib]"
```

## Quick example

```python
import numpy as np

from solardss.constraints import apply_exclusions
from solardss.indicators import minmax_normalize
from solardss.suitability import weighted_linear_combination

slope_score = minmax_normalize(
    np.array([2.0, 8.0, 15.0]),
    polarity="cost",
    range_min=0.0,
    range_max=15.0,
)
ghi_score = minmax_normalize(
    np.array([1600.0, 1800.0, 1500.0]),
    polarity="benefit",
    range_min=1400.0,
    range_max=2000.0,
)
suitability = weighted_linear_combination(
    {"slope": slope_score.values, "ghi": ghi_score.values},
    {"slope": 0.4, "ghi": 0.6},
)
excluded = apply_exclusions({"water": np.array([False, True, False])})
```

## Development

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
./.venv/bin/ruff format --check .
./.venv/bin/mypy src tests
```

## License

MIT. See [LICENSE](LICENSE).
