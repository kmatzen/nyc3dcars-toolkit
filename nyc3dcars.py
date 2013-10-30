import warnings
from sqlalchemy import exc as sa_exc

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import *

from datetime import datetime
import os
import ConfigParser
from collections import namedtuple

import numpy

warnings.simplefilter("ignore", category=sa_exc.SAWarning)

config = ConfigParser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'nyc3dcars.cfg'))
username = config.get('database', 'username')
password = config.get('database', 'password')
dbname, = config.get('database', 'name'),
host, = config.get('database', 'host'),
port = config.getint('database', 'port')
echo = config.getboolean('database', 'echo')

photo_dir = config.get('directories', 'images-dir')

engine = create_engine(
    'postgresql://%s:%s@%s:%d/%s' % (username, password, host, port, dbname),
    echo=echo,
)
metadata = MetaData(engine)
Base = declarative_base(metadata=metadata)

class VehicleType(Base):
    __tablename__ = 'vehicletypes'
    __table_args__ = {'autoload':True}
    detections = relationship('Detection', backref='type')
    vehicles = relationship('Vehicle', backref='type')

class Detection(Base):
    __tablename__ = 'detections'
    __table_args__ = {'autoload':True}

class Photo(Base):
    __tablename__ = 'photos'
    __table_args__ = {'autoload':True, 'extend_existing':True}
    lla = deferred(Column('lla', Binary))
    geom = deferred(Column('geom', Binary))
    vehicles = relationship('Vehicle', backref='photo')
    detections = relationship('Detection', backref='photo')

class Vehicle(Base):
    __tablename__ = 'vehicles'
    __table_args__ = {'autoload':True}

class Model(Base):
    __tablename__ = 'models'
    __table_args__ = {'autoload':True}
    detections = relationship('Detection', backref='model')

class OsmLine(Base):
    __tablename__ = 'planet_osm_line'
    __table_args__ = {'autoload': True}
    osm_id = Column(Integer, primary_key=True)

class Footprint(Base):
    __tablename__ = 'footprints'
    __table_args__ = {'autoload': True}

class GeoidHeight(Base):
    __tablename__ = 'geoidheights'
    __table_args__ = {'autoload': True}

class Median(Base):
    __tablename__ = 'medians'
    __table_args__ = {'autoload': True}

class Roadbed(Base):
    __tablename__ = 'roadbeds'
    __table_args__ = {'autoload': True}

class Sidewalk(Base):
    __tablename__ = 'sidewalks'
    __table_args__ = {'autoload': True}

class ElevationRaster(Base):
    __tablename__ = 'elevation'
    __table_args__ = {'autoload': True}

Base.metadata.create_all(engine)
Session = scoped_session(sessionmaker(bind=engine))

"""
from sqlalchemy import event
from sqlalchemy.engine import Engine
import time
import logging
logger = logging.getLogger("myapp.sqltime")
logger.setLevel(logging.DEBUG)

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, 
                        parameters, context, executemany):
    context._query_start_time = time.time()
    logger.debug("Start Query: %s" % statement)

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, 
                        parameters, context, executemany):
    total = time.time() - context._query_start_time
    logger.debug("Query Complete!")
    logger.debug("Total Time: %f" % total)
"""
