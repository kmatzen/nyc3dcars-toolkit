import warnings
from sqlalchemy import exc as sa_exc

from sqlalchemy import create_engine, MetaData, Column, Binary, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, deferred, scoped_session, sessionmaker

import os
import ConfigParser

warnings.simplefilter("ignore", category=sa_exc.SAWarning)

def init ():
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'nyc3dcars.cfg'))
    username = config.get('database', 'username')
    password = config.get('database', 'password')
    dbname, = config.get('database', 'name'),
    host, = config.get('database', 'host'),
    port = config.getint('database', 'port')
    echo = config.getboolean('database', 'echo')

    engine = create_engine(
        'postgresql://%s:%s@%s:%d/%s' % (username, password, host, port, dbname),
        echo=echo,
    )

    return engine

ENGINE = init()

METADATA = MetaData(ENGINE)
BASE = declarative_base(metadata=METADATA)

class VehicleType(BASE):
    __tablename__ = 'vehicletypes'
    __table_args__ = {'autoload':True}
    detections = relationship('Detection', backref='type')
    vehicles = relationship('Vehicle', backref='type')

class Detection(BASE):
    __tablename__ = 'detections'
    __table_args__ = {'autoload':True}

class Photo(BASE):
    __tablename__ = 'photos'
    __table_args__ = {'autoload':True, 'extend_existing':True}
    lla = deferred(Column('lla', Binary))
    geom = deferred(Column('geom', Binary))
    vehicles = relationship('Vehicle', backref='photo')
    detections = relationship('Detection', backref='photo')

class Vehicle(BASE):
    __tablename__ = 'vehicles'
    __table_args__ = {'autoload':True}

class Model(BASE):
    __tablename__ = 'models'
    __table_args__ = {'autoload':True}
    detections = relationship('Detection', backref='model')

class OsmLine(BASE):
    __tablename__ = 'planet_osm_line'
    __table_args__ = {'autoload': True}
    osm_id = Column(Integer, primary_key=True)

class Footprint(BASE):
    __tablename__ = 'footprints'
    __table_args__ = {'autoload': True}

class GeoidHeight(BASE):
    __tablename__ = 'geoidheights'
    __table_args__ = {'autoload': True}

class Median(BASE):
    __tablename__ = 'medians'
    __table_args__ = {'autoload': True}

class Roadbed(BASE):
    __tablename__ = 'roadbeds'
    __table_args__ = {'autoload': True}

class Sidewalk(BASE):
    __tablename__ = 'sidewalks'
    __table_args__ = {'autoload': True}

class ElevationRaster(BASE):
    __tablename__ = 'elevation'
    __table_args__ = {'autoload': True}

BASE.metadata.create_all(ENGINE)
SESSION = scoped_session(sessionmaker(bind=ENGINE))

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
