# pylint: disable=W0232,R0903
# Most of these classes are fine as-is since sqlalchemy will modify them.

"""Manages access to nyc3dcars database."""

import warnings
from sqlalchemy import exc as sa_exc

from sqlalchemy import create_engine, MetaData, Column, Binary, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, deferred, scoped_session, sessionmaker

import os
import ConfigParser

warnings.simplefilter("ignore", category=sa_exc.SAWarning)

__config__ = ConfigParser.ConfigParser()
__config__.read(os.path.join(os.path.dirname(__file__), 'nyc3dcars.cfg'))
__username__ = __config__.get('database', 'username')
__password__ = __config__.get('database', 'password')
__dbname__ = __config__.get('database', 'name')
__host__ = __config__.get('database', 'host')
__port__ = __config__.getint('database', 'port')
__echo__ = __config__.getboolean('database', 'echo')

IMAGE_DIR = __config__.get('directories', 'image-dir')

__engine__ = create_engine(
    'postgresql://%s:%s@%s:%d/%s' % (
        __username__,
        __password__,
        __host__,
        __port__,
        __dbname__
    ),
    echo=__echo__,
)

__metadata__ = MetaData(__engine__)
__Base__ = declarative_base(metadata=__metadata__)


class VehicleType(__Base__):

    """Sedan, minivan, etc."""

    __tablename__ = 'vehicle_types'
    __table_args__ = {'autoload': True}
    detections = relationship('Detection', backref='vehicle_type')
    vehicles = relationship('Vehicle', backref='vehicle_type')


class Detection(__Base__):

    """DPM detection result."""

    __tablename__ = 'detections'
    __table_args__ = {'autoload': True}


class Photo(__Base__):

    """Photograph along with geo-location data."""

    __tablename__ = 'photos'
    __table_args__ = {'autoload': True, 'extend_existing': True}
    lla = deferred(Column('lla', Binary))
    geom = deferred(Column('geom', Binary))
    vehicles = relationship('Vehicle', backref='photo')
    detections = relationship('Detection', backref='photo')


class Dataset(__Base__):

    """Contains the reconstruction correction factors."""

    __tablename__ = 'datasets'
    __table_args__ = {'autoload': True}

    photos = relationship('Photo', backref='dataset')


class Vehicle(__Base__):

    """Ground truth vehicle annotation."""

    __tablename__ = 'vehicles'
    __table_args__ = {'autoload': True}


class Model(__Base__):

    """Registered DPM model."""

    __tablename__ = 'models'
    __table_args__ = {'autoload': True}
    detections = relationship('Detection', backref='model')


class PlanetOsmLine(__Base__):

    """OpenStreetMap polylines.  Roads, railways, etc."""

    __tablename__ = 'planet_osm_line'
    __table_args__ = {'autoload': True}
    osm_id = Column(Integer, primary_key=True)


class Footprint(__Base__):

    """Building footprints from OpenData."""

    __tablename__ = 'footprints'
    __table_args__ = {'autoload': True}


class GeoidHeight(__Base__):

    """Distance used to turn ellipsoid height into orthometric height."""

    __tablename__ = 'geoid_heights'
    __table_args__ = {'autoload': True}


class Median(__Base__):

    """Road medians from OpenData."""

    __tablename__ = 'medians'
    __table_args__ = {'autoload': True}


class Roadbed(__Base__):

    """Roadbed polygons from OpenData."""

    __tablename__ = 'roadbeds'
    __table_args__ = {'autoload': True}


class Sidewalk(__Base__):

    """Sidewalk polygons from OpenData."""

    __tablename__ = 'sidewalks'
    __table_args__ = {'autoload': True}


class Elevation(__Base__):

    """Terrain elevation raster from USGS."""

    __tablename__ = 'elevations'
    __table_args__ = {'autoload': True}

__Base__.metadata.create_all(__engine__)
SESSION = scoped_session(sessionmaker(bind=__engine__))
