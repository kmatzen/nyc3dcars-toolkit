"""Implements all of the functions that compute the geographic
   scores using geodatabase."""

from nyc3dcars import Photo, Detection, Elevation, \
    PlanetOsmLine, Roadbed, GeoidHeight
from sqlalchemy import func

import sys
from collections import namedtuple
import numpy
import math
import logging

import pygeo

import gdal
from gdalconst import GA_ReadOnly

__ALT_DIFF_CACHE__ = {}
__HORIZON_CACHE__ = {}
__ELEVATION_RASTER__ = None
__GEOIDHEIGHT_RASTER__ = None


def read_elevation_raster(session):
    """Reads the entire elevation raster out of the db and saves it locally.
       postgis is very slow for random access to rasters right now."""

    name = 'elevation-cached.tif'

    dataset = gdal.Open(name, GA_ReadOnly)
    if dataset is not None:
        logging.info('elevation raster loaded from cached file')
    else:
        logging.info('building elevation raster from nyc3dcars')
        # pylint: disable-msg=E1101
        union = func.ST_Union(Elevation.rast)
        # pylint: enable-msg=E1101
        gtiff = func.ST_AsGDALRaster(union, 'GTiff')

        raster, = session.query(gtiff) \
            .one()

        with open(name, 'wb') as raster_file:
            raster_file.write(raster)

        dataset = gdal.Open(name, GA_ReadOnly)

    return parse_dataset(dataset)


def read_geoidheights_raster(session):
    """Reads the entire geoidheight raster out of the db and saves it locally.
       postgis is very slow for random access to rasters right now."""

    name = 'geoidheight-cached.tif'

    dataset = gdal.Open(name, GA_ReadOnly)
    if dataset is not None:
        logging.info('geoidheight raster loaded from cached file')
    else:
        logging.info('building geoidheight raster from nyc3dcars')
        # pylint: disable-msg=E1101
        union = func.ST_Union(GeoidHeight.rast)
        # pylint: enable-msg=E1101
        gtiff = func.ST_AsGDALRaster(union, 'GTiff')

        raster, = session.query(gtiff) \
            .one()

        with open(name, 'wb') as raster_file:
            raster_file.write(raster)

        dataset = gdal.Open(name, GA_ReadOnly)

    return parse_dataset(dataset)


def parse_dataset(dataset):
    """Builds an easier to use dataset tuple out of the GDAL dataset."""

    dataset_tuple = namedtuple('dataset_tuple', 'data, x, y, width, height')

    geotransform = dataset.GetGeoTransform()
    return dataset_tuple(
        data=dataset.ReadAsArray(),
        x=geotransform[0],
        y=geotransform[3],
        width=geotransform[1],
        height=geotransform[5]
    )


def index_raster(dataset, lat, lon):
    """Index into raster using lat and long."""

    lat_idx = (lat - dataset.y) / dataset.height
    lon_idx = (lon - dataset.x) / dataset.width
    try:
        return dataset.data[lat_idx, lon_idx]
    except IndexError:
        return numpy.inf


def roadbed_query(session, detection):
    """Find roadbeds that intersect the detection's footprint."""

    car_lla = detection.lonlat

    # pylint: disable-msg=E1101
    roadbeds4326 = func.ST_Transform(Roadbed.geom, 4326)
    car_roadbed_dist = func.ST_Distance(roadbeds4326, car_lla)

    query = session.query(
        car_roadbed_dist,
        Roadbed.gid) \
        .filter(func.ST_Intersects(car_lla, roadbeds4326)) \
        .order_by(car_roadbed_dist.asc())
    # pylint: enable-msg=E1101
    roadbed = query.first()
    return roadbed


def centerline_query(session, detection):
    """Finds the centerline orientation that most closely agrees with
       detection-intersected roadbeds."""

    # pylint: disable-msg=E1101
    car_polygon = Detection.geom
    car_polygon102718 = func.ST_Transform(car_polygon, 102718)
    car_filter = func.ST_Intersects(
        Roadbed.geom,
        car_polygon102718
    )

    query = session.query(
        Roadbed.gid) \
        .filter(Detection.id == detection.id) \
        .filter(car_filter)
    road_gids = query.all()

    if len(road_gids) == 0:
        return

    lat, lon, alt = session.query(
        func.ST_Y(Detection.lla),
        func.ST_X(Detection.lla),
        func.ST_Z(Detection.lla)) \
        .filter(Detection.id == detection.id) \
        .one()
    lla = numpy.array([[lat, lon, alt]])
    enu = pygeo.LLAToENU(lla).reshape((3, 3))

    roadbeds4326 = func.ST_Transform(Roadbed.geom, 4326)

    centerlines4326 = PlanetOsmLine.way
    centerline_filter = func.ST_Intersects(roadbeds4326, centerlines4326)
    centerline_frac = func.ST_Line_Locate_Point(
        centerlines4326, Detection.lla)
    centerline_start_frac = func.least(1, centerline_frac + 0.01)
    centerline_end_frac = func.greatest(0, centerline_frac - 0.01)
    centerline_start = func.ST_Line_Interpolate_Point(centerlines4326,
                                                      centerline_start_frac)
    centerline_end = func.ST_Line_Interpolate_Point(centerlines4326,
                                                    centerline_end_frac)

    segments = session.query(
        func.ST_Y(centerline_start).label('lats'),
        func.ST_X(centerline_start).label('lons'),

        func.ST_Y(centerline_end).label('late'),
        func.ST_X(centerline_end).label('lone'),

        PlanetOsmLine.oneway) \
        .filter(Detection.id == detection.id) \
        .filter(centerline_filter) \
        .filter(Roadbed.gid.in_(road_gids)) \
        .filter(PlanetOsmLine.osm_id >= 0) \
        .filter(PlanetOsmLine.railway.__eq__(None))
    # pylint: enable-msg=E1101

    for segment in segments:
        segment_start = pygeo.LLAToECEF(numpy.array(
            [[segment.lats, segment.lons, alt]],
            dtype=numpy.float64
        ))
        segment_end = pygeo.LLAToECEF(numpy.array(
            [[segment.late, segment.lone, alt]],
            dtype=numpy.float64
        ))

        segment_dir = (segment_end - segment_start)
        segment_dir /= numpy.linalg.norm(segment_dir)

        segment_rot = enu.T.dot(segment_dir.T)

        segment_angle = math.atan2(segment_rot[1], segment_rot[0])

        yield segment_angle, segment.oneway


def elevation_query(session, detection, elevation_raster, geoidheight_raster):
    """Computes the elevation of the detection above the terrain."""

    # pylint: disable-msg=E1101
    car_lla = Detection.lla
    query = session.query(
        func.ST_Y(car_lla),
        func.ST_X(car_lla),
        func.ST_Z(car_lla)) \
        .filter(Detection.id == detection.id)
    # pylint: enable-msg=E1101
    lat, lon, alt = query.one()

    elevation = index_raster(elevation_raster, lat, lon)
    geoidheight = index_raster(
        geoidheight_raster, lat, lon if lon > 0 else lon + 360)

    return alt - elevation - geoidheight


def coverage_query(session, detection):
    """Computes the percentage of the vehicles on the roadbeds."""

    # pylint: disable-msg=E1101
    car_polygon = Detection.geom
    car_polygon102718 = func.ST_Transform(car_polygon, 102718)
    car_road_intersection = func.ST_Area(
        func.ST_Intersection(Roadbed.geom, car_polygon102718))
    car_area = func.ST_Area(car_polygon102718)
    car_filter = func.ST_Intersects(
        Roadbed.geom,
        car_polygon102718)

    query = session.query(
        func.sum(car_road_intersection / car_area)) \
        .filter(Detection.id == detection.id) \
        .filter(car_filter)
    # pylint: enable-msg=E1101
    coverage, = query.one()
    if coverage is None:
        coverage = 0
    return coverage


def centerline_angle_diff(detection, centerline_angle, oneway):
    """Computes the angle between the vehicle orientation
       and the expected directions of travel."""

    ex_cam_angle = detection.world_angle - math.pi / 2
    diff = math.acos(math.cos(centerline_angle - ex_cam_angle))
    twoway_types = ('undefined', 'reversible', 'yes; no', '-1', 'no', None)
    if oneway in twoway_types:
        new_diff = math.acos(
            math.cos(math.pi + centerline_angle - ex_cam_angle))
        if new_diff < diff:
            diff = new_diff
    return diff


def get_horizon_endpoints(session, photo):
    """Computes the endpoints of the horizon in the photo."""

    if photo.id in __HORIZON_CACHE__:
        return __HORIZON_CACHE__[photo.id]

    lon, lat, alt = session.query(
        func.ST_X(Photo.lla),
        func.ST_Y(Photo.lla),
        func.ST_Z(Photo.lla)) \
        .filter_by(id=photo.id) \
        .one()

    point = numpy.array([[lat, lon, alt]])

    enu = pygeo.LLAToENU(point).reshape((3, 3))

    R = numpy.array([
        [photo.r11, photo.r12, photo.r13],
        [photo.r21, photo.r22, photo.r23],
        [photo.r31, photo.r32, photo.r33],
    ])

    K = numpy.array([
        [photo.focal,            0,  photo.width / 2],
        [0,  photo.focal, photo.height / 2],
        [0,            0,              1],
    ])

    P = K.dot(R.dot(enu))

    h = numpy.cross(P[:, 0], P[:, 1])

    m = -h[0] / h[1]
    b = -h[2] / h[1]

    endpoints = numpy.array([
        [0, (m * photo.width + b) / photo.height],
        [1, b / photo.height],
    ])

    __HORIZON_CACHE__[photo.id] = endpoints
    return __HORIZON_CACHE__[photo.id]


def score_horizon(session, detection):
    """Scores detection based on whether or not it sits above the horizon."""

    endpoints = get_horizon_endpoints(session, detection.photo)

    Ax = endpoints[0, 0]
    Ay = endpoints[0, 1]
    Bx = endpoints[1, 0]
    By = endpoints[1, 1]

    Cx1 = detection.x1
    Cy1 = detection.y2
    Cx2 = detection.x2
    Cy2 = detection.y2

    score1 = (Bx - Ax) * (Cy1 - Ay) - (By - Ay) * (Cx1 - Ax)
    score2 = (Bx - Ax) * (Cy2 - Ay) - (By - Ay) * (Cx2 - Ax)

    return 1 if score1 > 0 and score2 > 0 else 0


def get_alt_diff(session, detection):
    """Caches elevation score results."""

    # pylint: disable-msg=W0603
    global __ELEVATION_RASTER__
    global __GEOIDHEIGHT_RASTER__
    # pylint: enable-msg=W0603

    if detection.id in __ALT_DIFF_CACHE__:
        return __ALT_DIFF_CACHE__[detection.id]

    if __ELEVATION_RASTER__ is None:
        logging.info('loading elevation raster')
        __ELEVATION_RASTER__ = read_elevation_raster(session)

    if __GEOIDHEIGHT_RASTER__ is None:
        logging.info('loading geoidheight raster')
        __GEOIDHEIGHT_RASTER__ = read_geoidheights_raster(session)

    __ALT_DIFF_CACHE__[detection.id] = elevation_query(
        session,
        detection,
        __ELEVATION_RASTER__,
        __GEOIDHEIGHT_RASTER__
    )
    return __ALT_DIFF_CACHE__[detection.id]


def elevation_score(session, detection, sigma):
    """Computes elevation score."""

    elevation_diff = get_alt_diff(session, detection)

    score = math.exp(-0.5 * (elevation_diff / sigma) ** 2)
    if score < sys.float_info.min:
        return sys.float_info.min
    return score


def get_orientation_error(session, detection):
    """Computes angle error."""

    angle_centerlines = centerline_query(session, detection)

    diff = math.pi
    for angle, oneway in angle_centerlines:
        newdiff = centerline_angle_diff(detection, angle, oneway)
        if newdiff < diff:
            diff = newdiff

    return diff


def orientation_score_continuous(session, detection):
    """Angle error which does not consider the discritized DPM viewpoints."""

    orientation_error = get_orientation_error(session, detection)

    sigma_angle = math.pi / 12
    angle_score = math.exp(-0.5 * (orientation_error / sigma_angle) ** 2)
    if math.fabs(angle_score) < sys.float_info.min:
        return sys.float_info.min
    return angle_score


def orientation_score_discrete(session, detection):
    """Angle error which does consider the discritized DPM viewpoints."""

    orientation_error = get_orientation_error(session, detection)

    if orientation_error < math.radians(11.25):
        return 1.0
    elif orientation_error < math.radians(33.75):
        return 0.5
    else:
        return sys.float_info.min

SCORE = namedtuple('SCORE', 'name, compute, output')

# pylint: disable-msg=E1101
__Scores__ = [
    SCORE(
        name='prob',
        compute=None,
        output=Detection.prob,
    ),

    SCORE(
        name='coverage_score',
        compute=coverage_query,
        output=Detection.coverage_score,
    ),

    SCORE(
        name='height_score',
        compute=lambda s, d: elevation_score(s, d, math.sqrt(2.44)),
        output=Detection.height_score,
    ),

    SCORE(
        name='height1_score',
        compute=lambda s, d: elevation_score(s, d, 1),
        output=Detection.height1_score,
    ),

    SCORE(
        name='height2_score',
        compute=lambda s, d: elevation_score(s, d, 0.5),
        output=Detection.height2_score,
    ),

    SCORE(
        name='height3_score',
        compute=lambda s, d: elevation_score(s, d, 5),
        output=Detection.height3_score,
    ),

    SCORE(
        name='height4_score',
        compute=lambda s, d: elevation_score(s, d, 10),
        output=Detection.height4_score,
    ),

    SCORE(
        name='height5_score',
        compute=lambda s, d: elevation_score(s, d, 20),
        output=Detection.height5_score,
    ),

    SCORE(
        name='height6_score',
        compute=lambda s, d: elevation_score(s, d, 50),
        output=Detection.height6_score,
    ),

    SCORE(
        name='height7_score',
        compute=lambda s, d: elevation_score(s, d, 100),
        output=Detection.height7_score,
    ),

    SCORE(
        name='angle_score',
        compute=orientation_score_continuous,
        output=Detection.angle_score,
    ),

    SCORE(
        name='angle2_score',
        compute=orientation_score_discrete,
        output=Detection.angle2_score,
    ),

    SCORE(
        name='horizon_score',
        compute=score_horizon,
        output=Detection.horizon_score,
    ),
]
# pylint: enable-msg=E1101

SCORES = {s.name: s for s in __Scores__}

METHOD = namedtuple('METHOD', 'name, score, inputs, output, display')

# pylint: disable-msg=E1101
__Methods__ = [
    METHOD(
        name='reference',
        score=Detection.prob,
        inputs=[
            Detection.prob,
        ],
        output=Detection.nms,
        display=True,
    ),

    METHOD(
        name='coverage',
        score=Detection.prob * Detection.coverage_score,
        inputs=[
            Detection.prob,
            Detection.coverage_score,
        ],
        output=Detection.coverage_nms,
        display=True,
    ),

    METHOD(
        name='angle',
        score=Detection.prob * Detection.angle_score,
        inputs=[
            Detection.prob,
            Detection.angle_score,
        ],
        output=Detection.angle_nms,
        display=False,
    ),

    METHOD(
        name='angle2',
        score=Detection.prob * Detection.angle2_score,
        inputs=[
            Detection.prob,
            Detection.angle2_score,
        ],
        output=Detection.angle2_nms,
        display=True,
    ),

    METHOD(
        name='height',
        score=Detection.prob * Detection.height_score,
        inputs=[
            Detection.prob,
            Detection.height_score,
        ],
        output=Detection.height_nms,
        display=False,
    ),

    METHOD(
        name='height1',
        score=Detection.prob * Detection.height1_score,
        inputs=[
            Detection.prob,
            Detection.height1_score,
        ],
        output=Detection.height1_nms,
        display=False,
    ),

    METHOD(
        name='height2',
        score=Detection.prob * Detection.height2_score,
        inputs=[
            Detection.prob,
            Detection.height2_score,
        ],
        output=Detection.height2_nms,
        display=True,
    ),

    METHOD(
        name='height3',
        score=Detection.prob * Detection.height3_score,
        inputs=[
            Detection.prob,
            Detection.height3_score,
        ],
        output=Detection.height3_nms,
        display=False,
    ),

    METHOD(
        name='height4',
        score=Detection.prob * Detection.height4_score,
        inputs=[
            Detection.prob,
            Detection.height4_score,
        ],
        output=Detection.height4_nms,
        display=False,
    ),

    METHOD(
        name='height5',
        score=Detection.prob * Detection.height5_score,
        inputs=[
            Detection.prob,
            Detection.height5_score,
        ],
        output=Detection.height5_nms,
        display=False,
    ),

    METHOD(
        name='height6',
        score=Detection.prob * Detection.height6_score,
        inputs=[
            Detection.prob,
            Detection.height6_score,
        ],
        output=Detection.height6_nms,
        display=False,
    ),

    METHOD(
        name='height7',
        score=Detection.prob * Detection.height7_score,
        inputs=[
            Detection.prob,
            Detection.height7_score,
        ],
        output=Detection.height7_nms,
        display=False,
    ),

    METHOD(
        name='angle_height',
        score=Detection.prob *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.angle_score
        ) *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.height_score
        ),
        inputs=[
            Detection.prob,
            Detection.height_score,
            Detection.angle_score,
        ],
        output=Detection.angle_height_nms,
        display=False,
    ),

    METHOD(
        name='angle2_height',
        score=Detection.prob *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.angle2_score
        ) *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.height_score
        ),
        inputs=[
            Detection.prob,
            Detection.height_score,
            Detection.angle2_score,
        ],
        output=Detection.angle2_height_nms,
        display=False,
    ),

    METHOD(
        name='horizon',
        score=Detection.prob * Detection.horizon_score,
        inputs=[
            Detection.prob,
            Detection.horizon_score,
        ],
        output=Detection.horizon_nms,
        display=True,
    ),

    METHOD(
        name='all',
        score=Detection.prob *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.height_score
        ) *
        Detection.coverage_score *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.angle2_score
        ),
        inputs=[
            Detection.prob,
            Detection.height_score,
            Detection.angle_score,
            Detection.coverage_score,
        ],
        output=Detection.all_nms,
        display=False,
    ),

    METHOD(
        name='all2',
        score=Detection.prob *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.height2_score
        ) *
        Detection.coverage_score *
        func.greatest(
            math.sqrt(sys.float_info.min),
            Detection.angle2_score
        ),
        inputs=[
            Detection.prob,
            Detection.height2_score,
            Detection.angle2_score,
            Detection.coverage_score,
        ],
        output=Detection.all2_nms,
        display=True,
    ),
]
# pylint: enable-msg=E1101

METHODS = {m.name: m for m in __Methods__}
