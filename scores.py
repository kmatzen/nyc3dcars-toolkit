import nyc3dcars
from sqlalchemy import *

import sys
from collections import namedtuple
import numpy
import math
import logging

import pygeo

import gdal
from gdalconst import *

alt_diff_cache = {}
horizon_cache = {}
elevation_raster = None
geoidheight_raster = None

def read_elevation_raster(session):
    name = 'elevation-cached.tif'

    dataset = gdal.Open(name, GA_ReadOnly)
    if dataset is not None:
        logging.info('elevation raster loaded from cached file')
    else:
        logging.info('building elevation raster from nyc3dcars')
        union = func.ST_Union(nyc3dcars.ElevationRaster.rast)
        gtiff = func.ST_AsGDALRaster(union, 'GTiff')

        raster, = session.query(gtiff) \
            .one()

        with open(name, 'wb') as fd:
            fd.write(raster)

        dataset = gdal.Open(name, GA_ReadOnly)

    return parseDataset(dataset)


def read_geoidheights_raster(session):
    name = 'geoidheight-cached.tif'

    dataset = gdal.Open(name, GA_ReadOnly)
    if dataset is not None:
        logging.info('geoidheight raster loaded from cached file')
    else:
        logging.info('building geoidheight raster from nyc3dcars')
        union = func.ST_Union(nyc3dcars.GeoidHeight.rast)
        gtiff = func.ST_AsGDALRaster(union, 'GTiff')

        raster, = session.query(gtiff) \
            .one()

        with open(name, 'wb') as fd:
            fd.write(raster)

        dataset = gdal.Open(name, GA_ReadOnly)

    return parseDataset(dataset)

Dataset = namedtuple('Dataset', 'data, x, y, width, height')


def parseDataset(dataset):
    geotransform = dataset.GetGeoTransform()
    return Dataset(
        data=dataset.ReadAsArray(),
        x=geotransform[0],
        y=geotransform[3],
        width=geotransform[1],
        height=geotransform[5]
    )


def index_raster(dataset, lat, lon):
    lat_idx = (lat - dataset.y) / dataset.height
    lon_idx = (lon - dataset.x) / dataset.width
    try:
        return dataset.data[lat_idx, lon_idx]
    except IndexError:
        return float('Infinity')


def roadbed_query(session, detection):
    photo = detection.photo
    car_lla = detection.lonlat

    roadbeds4326 = func.ST_Transform(nyc3dcars.Roadbed.geom, 4326)
    car_roadbed_dist = func.ST_Distance(roadbeds4326, car_lla)

    query = session.query(
        car_roadbed_dist,
        nyc3dcars.Roadbed.gid) \
        .filter(func.ST_Intersects(car_lla, roadbeds4326)) \
        .order_by(car_roadbed_dist.asc())
    roadbed = query.first()
    return roadbed


def centerline_query(session, detection):
    ex_cam_angle = nyc3dcars.Detection.world_angle
    car_polygon = nyc3dcars.Detection.geom
    car_polygon102718 = func.ST_Transform(car_polygon, 102718)
    car_filter = func.ST_Intersects(
        nyc3dcars.Roadbed.geom,
        car_polygon102718)

    query = session.query(
        nyc3dcars.Roadbed.gid) \
        .filter(nyc3dcars.Detection.id == detection.id) \
        .filter(car_filter)
    road_gids = query.all()

    if len(road_gids) == 0:
        return 

    lat, lon, alt = session.query(
        func.ST_Y(nyc3dcars.Detection.lla),
        func.ST_X(nyc3dcars.Detection.lla),
        func.ST_Z(nyc3dcars.Detection.lla)) \
        .filter(nyc3dcars.Detection.id == detection.id) \
        .one()
    lla = numpy.array([[lat, lon, alt]])
    enu = pygeo.LLAToENU(lla).reshape((3,3))

    roadbeds4326 = func.ST_Transform(nyc3dcars.Roadbed.geom, 4326)

    centerlines4326 = nyc3dcars.OsmLine.way
    centerline_filter = func.ST_Intersects(roadbeds4326, centerlines4326)
    car_centerline_dist = func.ST_Distance(nyc3dcars.Detection.lla, centerlines4326)
    centerline_frac = func.ST_Line_Locate_Point(centerlines4326, nyc3dcars.Detection.lla)
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
        
        nyc3dcars.OsmLine.oneway) \
        .filter(nyc3dcars.Detection.id == detection.id) \
        .filter(centerline_filter) \
        .filter(nyc3dcars.Roadbed.gid.in_(road_gids)) \
        .filter(nyc3dcars.OsmLine.osm_id >= 0) \
        .filter(nyc3dcars.OsmLine.railway.__eq__(None))


    for segment in segments:
        segment_start = pygeo.LLAToECEF(numpy.array([[segment.lats, segment.lons, alt]], dtype=numpy.float64))
        segment_end = pygeo.LLAToECEF(numpy.array([[segment.late, segment.lone, alt]], dtype=numpy.float64))

        segment_dir = (segment_end - segment_start)
        segment_dir /= numpy.linalg.norm(segment_dir)

        segment_rot = enu.T.dot(segment_dir.T)

        segment_angle = math.atan2(segment_rot[1], segment_rot[0])

        yield segment_angle, segment.oneway

def elevation_query(session, detection, elevation_raster, geoidheight_raster):
    car_lla = nyc3dcars.Detection.lla
    query = session.query(
        func.ST_Y(car_lla),
        func.ST_X(car_lla),
        func.ST_Z(car_lla)) \
        .filter(nyc3dcars.Detection.id == detection.id)
    lat, lon, alt = query.one()

    elevation = index_raster(elevation_raster, lat, lon)
    geoidheight = index_raster(geoidheight_raster, lat, lon if lon > 0 else lon + 360)

    return alt - elevation - geoidheight

def coverage_query(session, detection):
    ex_cam_angle = nyc3dcars.Detection.world_angle
    car_polygon = nyc3dcars.Detection.geom
    car_polygon102718 = func.ST_Transform(car_polygon, 102718)
    car_road_intersection = func.ST_Area(
        func.ST_Intersection(nyc3dcars.Roadbed.geom, car_polygon102718))
    car_area = func.ST_Area(car_polygon102718)
    car_filter = func.ST_Intersects(
        nyc3dcars.Roadbed.geom,
        car_polygon102718)

    query = session.query(
        func.sum(car_road_intersection / car_area)) \
        .filter(nyc3dcars.Detection.id == detection.id) \
        .filter(car_filter)
    coverage, = query.one()
    if coverage is None:
        coverage = 0
    return coverage

def centerline_angle_diff(detection, centerline_angle, oneway):
    ex_cam_angle = detection.world_angle - math.pi/2
    diff = math.acos(math.cos(centerline_angle - ex_cam_angle)) 
    twoway_types = ('undefined', 'reversible', 'yes; no', '-1', 'no', None)
    if oneway in twoway_types:
        newDiff = math.acos(math.cos(math.pi + centerline_angle - ex_cam_angle))
        if newDiff < diff:
            diff = newDiff
    return diff

def get_horizon_endpoints(session, photo):
    if photo.id in horizon_cache:
        return horizon_cache[photo.id]

    logging.info('computing horizon endpoints %d'%photo.id)

    lon, lat, alt = session.query(
        func.ST_X(nyc3dcars.Photo.lla),
        func.ST_Y(nyc3dcars.Photo.lla),
        func.ST_Z(nyc3dcars.Photo.lla)) \
        .filter_by(id=photo.id) \
        .one()

    point = numpy.array([[lat, lon, alt]])

    ENU = pygeo.LLAToENU(point).reshape((3,3))

    R = numpy.array([
        [photo.r11, photo.r12, photo.r13],
        [photo.r21, photo.r22, photo.r23],
        [photo.r31, photo.r32, photo.r33],
    ])

    K = numpy.array([
        [photo.focal,            0,  photo.width/2],
        [          0,  photo.focal, photo.height/2],
        [          0,            0,              1],
    ])

    P = K.dot(R.dot(ENU))

    h = numpy.cross(P[:,0], P[:,1])

    m = -h[0]/h[1]
    b = -h[2]/h[1]

    endpoints = numpy.array([
        [0, (m*photo.width+b)/photo.height],
        [1, b/photo.height],
    ])

    horizon_cache[photo.id] = endpoints
    return horizon_cache[photo.id]

def score_horizon(session, detection):
    endpoints = get_horizon_endpoints(session, detection.photo)

    Ax = endpoints[0,0]
    Ay = endpoints[0,1]
    Bx = endpoints[1,0]
    By = endpoints[1,1]
    
    Cx1 = detection.x1
    Cy1 = detection.y2
    Cx2 = detection.x2
    Cy2 = detection.y2

    score1 = (Bx-Ax)*(Cy1-Ay)-(By-Ay)*(Cx1-Ax)
    score2 = (Bx-Ax)*(Cy2-Ay)-(By-Ay)*(Cx2-Ax)

    return 1 if score1 > 0 and score2 > 0 else 0

def get_alt_diff(session, detection):
    global elevation_raster
    global geoidheight_raster

    if detection.id in alt_diff_cache:
        return alt_diff_cache[detection.id]

    if elevation_raster is None:
        logging.info('loading elevation raster')
        elevation_raster = read_elevation_raster(session)

    if geoidheight_raster is None:
        logging.info('loading geoidheight raster')
        geoidheight_raster = read_geoidheights_raster(session)

    alt_diff_cache[detection.id] = elevation_query(session, detection, elevation_raster, geoidheight_raster)
    return alt_diff_cache[detection.id]

def elevation_score (session, detection, sigma):
    elevation_diff = get_alt_diff(session, detection)

    score = math.exp(-0.5 * (elevation_diff / sigma) ** 2)
    if score < sys.float_info.min:
        return sys.float_info.min
    return score

def get_orientation_error(session, detection):
    angle_centerlines = centerline_query(session, detection)

    diff = math.pi
    for angle, oneway in angle_centerlines:
        newdiff = centerline_angle_diff(detection, angle, oneway)
        if newdiff < diff:
            diff = newdiff

    return diff

def orientation_score_continuous(session, detection):
    orientation_error = get_orientation_error(session, detection)

    sigma_angle = math.pi / 12
    angle_score = math.exp(-0.5 * (orientation_error / sigma_angle) ** 2)
    if math.fabs(angle_score) < sys.float_info.min:
        return sys.float_info.min
    return angle_score

def orientation_score_discrete(session, detection):
    orientation_error = get_orientation_error(session, detection)

    if orientation_error < math.radians(11.25):
        return 1.0
    elif orientation_error < math.radians(33.75):
        return 0.5
    else:
        return sys.float_info.min

Score = namedtuple('Score', 'name, compute, output')

Scores = {s.name:s for s in [
    Score(
        name='prob',
        compute=None,
        output=nyc3dcars.Detection.prob,
    ),

    Score(
        name='coverage_score',
        compute=coverage_query,
        output=nyc3dcars.Detection.coverage_score,
    ),

    Score(
        name='height_score',
        compute=lambda s, d: elevation_score(s, d, math.sqrt(2.44)),
        output=nyc3dcars.Detection.height_score,
    ),

    Score(
        name='height1_score',
        compute=lambda s, d: elevation_score(s, d, 1),
        output=nyc3dcars.Detection.height1_score,
    ),

    Score(
        name='height2_score',
        compute=lambda s, d: elevation_score(s, d, 0.5),
        output=nyc3dcars.Detection.height2_score,
    ),

    Score(
        name='height3_score',
        compute=lambda s, d: elevation_score(s, d, 5),
        output=nyc3dcars.Detection.height3_score,
    ),

    Score(
        name='height4_score',
        compute=lambda s, d: elevation_score(s, d, 10),
        output=nyc3dcars.Detection.height4_score,
    ),

    Score(
        name='height5_score',
        compute=lambda s, d: elevation_score(s, d, 20),
        output=nyc3dcars.Detection.height5_score,
    ),

    Score(
        name='height6_score',
        compute=lambda s, d: elevation_score(s, d, 50),
        output=nyc3dcars.Detection.height6_score,
    ),

    Score(
        name='height7_score',
        compute=lambda s, d: elevation_score(s, d, 100),
        output=nyc3dcars.Detection.height7_score,
    ),

    Score(
        name='angle_score',
        compute=orientation_score_continuous,
        output=nyc3dcars.Detection.angle_score,
    ),

    Score(
        name='angle2_score',
        compute=orientation_score_discrete,
        output=nyc3dcars.Detection.angle2_score,
    ),

    Score(
        name='horizon_score',
        compute=score_horizon,
        output=nyc3dcars.Detection.horizon_score,
    ),
]}

Method = namedtuple('Method', 'name, score, inputs, output, display')

Methods = {m.name:m for m in [
    Method(
        name='reference',
        score=nyc3dcars.Detection.prob,
        inputs=[
            nyc3dcars.Detection.prob,
        ],
        output=nyc3dcars.Detection.nms,
        display=True,
    ),

    Method(
        name='coverage',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.coverage_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.coverage_score,
        ],
        output=nyc3dcars.Detection.coverage_nms,
        display=True,
    ),

    Method(
        name='angle',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.angle_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.angle_score,
        ],
        output=nyc3dcars.Detection.angle_nms,
        display=False,
    ),

    Method(
        name='angle2',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.angle2_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.angle2_score,
        ],
        output=nyc3dcars.Detection.angle2_nms,
        display=True,
    ),

    Method(
        name='height',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height_score,
        ],
        output=nyc3dcars.Detection.height_nms,
        display=False,
    ),

    Method(
        name='height1',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height1_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height1_score,
        ],
        output=nyc3dcars.Detection.height1_nms,
        display=False,
    ),

    Method(
        name='height2',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height2_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height2_score,
        ],
        output=nyc3dcars.Detection.height2_nms,
        display=True,
    ),

    Method(
        name='height3',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height3_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height3_score,
        ],
        output=nyc3dcars.Detection.height3_nms,
        display=False,
    ),

    Method(
        name='height4',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height4_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height4_score,
        ],
        output=nyc3dcars.Detection.height4_nms,
        display=False,
    ),

    Method(
        name='height5',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height5_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height5_score,
        ],
        output=nyc3dcars.Detection.height5_nms,
        display=False,
    ),

    Method(
        name='height6',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height6_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height6_score,
        ],
        output=nyc3dcars.Detection.height6_nms,
        display=False,
    ),
 
    Method(
        name='height7',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.height7_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height7_score,
        ],
        output=nyc3dcars.Detection.height7_nms,
        display=False,
    ),

    Method(
        name='angle_height',
        score=nyc3dcars.Detection.prob*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.angle_score)*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.height_score),
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height_score,
            nyc3dcars.Detection.angle_score,
        ],
        output=nyc3dcars.Detection.angle_height_nms,
        display=False,
    ),

    Method(
        name='angle2_height',
        score=nyc3dcars.Detection.prob*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.angle2_score)*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.height_score),
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height_score,
            nyc3dcars.Detection.angle2_score,
        ],
        output=nyc3dcars.Detection.angle2_height_nms,
        display=False,
    ),

    Method(
        name='horizon',
        score=nyc3dcars.Detection.prob*nyc3dcars.Detection.horizon_score,
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.horizon_score,
        ],
        output=nyc3dcars.Detection.horizon_nms,
        display=True,
    ),

    Method(
        name='all',
        score=nyc3dcars.Detection.prob*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.height_score)*nyc3dcars.Detection.coverage_score*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.angle2_score),
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height_score,
            nyc3dcars.Detection.angle_score,
            nyc3dcars.Detection.coverage_score,
        ],
        output=nyc3dcars.Detection.all_nms,
        display=False,
    ),

    Method(
        name='all2',
        score=nyc3dcars.Detection.prob*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.height2_score)*nyc3dcars.Detection.coverage_score*func.greatest(math.sqrt(sys.float_info.min), nyc3dcars.Detection.angle2_score),
        inputs=[
            nyc3dcars.Detection.prob,
            nyc3dcars.Detection.height2_score,
            nyc3dcars.Detection.angle2_score,
            nyc3dcars.Detection.coverage_score,
        ],
        output=nyc3dcars.Detection.all2_nms,
        display=True,
    ),
]}