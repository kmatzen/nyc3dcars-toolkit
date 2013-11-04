#!/usr/bin/env python

"""Computes the raw detections using the DPM.
   Additionally, estimates 3D pose for each detection."""

import itertools
import os
import argparse
import logging
import math
from collections import namedtuple

from nyc3dcars import SESSION, Photo, Detection, Model, VehicleType, IMAGE_DIR
from sqlalchemy import func
from sqlalchemy.orm import joinedload

import numpy
import scipy.misc

from celery.task import task

import pygeo

import pydro.io
import pydro.features


def in_range(val, low, high):
    """Checks if angle is within a certain range."""

    low -= 1e-5
    high += 1e-5

    twopi = 2 * math.pi

    low = (low % twopi + twopi) % twopi
    val = (val % twopi + twopi) % twopi
    high = (high % twopi + twopi) % twopi

    while high < low:
        high += 2 * math.pi

    while val < low:
        val += 2 * math.pi

    return val < high


def compute_car_pose(photo, bbox, angle, vehicle_types):
    """Compute 3D pose for 2D bounding box."""

    camera_rotation = numpy.array([[photo.r11, photo.r12, photo.r13],
                                   [photo.r21, photo.r22, photo.r23],
                                   [photo.r31, photo.r32, photo.r33]])
    camera_position = - \
        camera_rotation.T.dot([[photo.t1], [photo.t2], [photo.t3]])

    # Small correction factor computed from NYC3DCars annotation results.
    dataset_correction = numpy.array([
        [photo.dataset.t1],
        [photo.dataset.t2],
        [photo.dataset.t3],
    ])

    camera_position += dataset_correction

    # Just approximate it for this first calculation and correct it later.
    vehicle_height = 1.445

    det_focal = photo.focal
    det_height = photo.height
    det_width = photo.width
    det_bottom = bbox.y2 * det_height
    det_top = bbox.y1 * det_height
    det_middle = (bbox.x1 + bbox.x2) / 2 * det_width

    new_dir = numpy.array([[(det_middle - det_width / 2) / det_focal],
                          [(det_height / 2 - det_bottom) / det_focal],
                          [-1]])

    distance = vehicle_height / ((det_height / 2 - det_top) / det_focal - (
        det_height / 2 - det_bottom) / det_focal)

    car_position_wrt_camera = distance * new_dir
    car_position = camera_rotation.T.dot(car_position_wrt_camera)

    car_ecef = car_position + camera_position
    car_lla = pygeo.ECEFToLLA(car_ecef.T)
    car_enu = pygeo.LLAToENU(car_lla).reshape((3, 3))

    middle_x = (bbox.x1 + bbox.x2) / 2
    middle_y = (bbox.y1 + bbox.y2) / 2

    left_ray = numpy.array(
        [[(bbox.x1 * photo.width - det_width / 2) / det_focal],
         [(det_height / 2 - middle_y * photo.height) / det_focal],
         [-1]])

    left_ray_enu = car_enu.T.dot(camera_rotation.T.dot(left_ray))

    right_ray = numpy.array(
        [[(bbox.x2 * photo.width - det_width / 2) / det_focal],
         [(det_height / 2 - middle_y * photo.height) / det_focal],
         [-1]])

    right_ray_enu = car_enu.T.dot(camera_rotation.T.dot(right_ray))

    middle_ray = numpy.array(
        [[(middle_x * photo.width - det_width / 2) / det_focal],
         [(det_height / 2 - middle_y * photo.height) / det_focal],
         [-1]])

    middle_ray_enu = car_enu.T.dot(camera_rotation.T.dot(middle_ray))

    top_ray = numpy.array(
        [[(middle_x * photo.width - det_width / 2) / det_focal],
         [(det_height / 2 - bbox.y1 * photo.height) / det_focal],
         [-1]])

    top_ray_enu = car_enu.T.dot(camera_rotation.T.dot(top_ray))

    bottom_ray = numpy.array(
        [[(middle_x * photo.width - det_width / 2) / det_focal],
         [(det_height / 2 - bbox.y2 * photo.height) / det_focal],
         [-1]])

    bottom_ray_enu = car_enu.T.dot(camera_rotation.T.dot(bottom_ray))

    middle_angle = math.atan2(middle_ray_enu[1], middle_ray_enu[0])
    right_angle = math.atan2(right_ray_enu[1], right_ray_enu[0])
    left_angle = math.atan2(left_ray_enu[1], left_ray_enu[0])

    if not angle:
        total_angle = middle_angle
    else:
        total_angle = middle_angle + angle

    for vehicle_type in vehicle_types:
        half_width = 0.3048 * vehicle_type.tight_width / 2
        half_length = 0.3048 * vehicle_type.tight_length / 2
        height = 0.3048 * vehicle_type.tight_height

        pointa = numpy.array([[half_width], [half_length]])
        pointb = numpy.array([[half_width],  [-half_length]])
        pointc = numpy.array([[-half_width], [-half_length]])
        pointd = numpy.array([[-half_width], [half_length]])

        half_pi = math.pi / 2

        if in_range(total_angle, right_angle, left_angle):
            left = pointd
            right = pointc

        elif in_range(total_angle, left_angle, half_pi + right_angle):
            left = pointa
            right = pointc

        elif in_range(total_angle, half_pi + right_angle, left_angle + half_pi):
            left = pointa
            right = pointd

        elif in_range(total_angle, left_angle + half_pi, right_angle + math.pi):
            left = pointb
            right = pointd

        elif in_range(total_angle, right_angle + math.pi, left_angle + math.pi):
            left = pointd
            right = pointa

        elif in_range(total_angle, left_angle + math.pi, 3 * half_pi + right_angle):
            left = pointc
            right = pointa

        elif in_range(total_angle, 3 * half_pi + right_angle, left_angle + 3 * half_pi):
            left = pointc
            right = pointb

        elif in_range(total_angle, left_angle + 3 * half_pi, right_angle):
            left = pointd
            right = pointb

        else:
            raise Exception('Invalid angle???')

        rot = numpy.array([
            [math.cos(total_angle), -math.sin(total_angle)],
            [math.sin(total_angle), math.cos(total_angle)],
        ])

        left_rot = rot.dot(left)
        right_rot = rot.dot(right)

        A = numpy.array([
            [left_ray_enu[1][0], -left_ray_enu[0][0]],
            [right_ray_enu[1][0], -right_ray_enu[0][0]],
        ])

        b = numpy.array([
            [-left_rot[0][0] * left_ray_enu[1][0]
                + left_rot[1][0] * left_ray_enu[0][0]],
            [-right_rot[0][0] * right_ray_enu[1][0]
                + right_rot[1][0] * right_ray_enu[0][0]],
        ])

        x = numpy.linalg.solve(A, b)

        a_rot = rot.dot(pointa)
        b_rot = rot.dot(pointb)
        c_rot = rot.dot(pointc)
        d_rot = rot.dot(pointd)

        distance = numpy.linalg.norm(x)
        bottom_point = distance * bottom_ray_enu / \
            numpy.linalg.norm(bottom_ray_enu)

        left_right_position = numpy.array([
            x[0],
            x[1],
            bottom_point[2],
        ])

        A = numpy.hstack((top_ray_enu, -bottom_ray_enu))
        b = numpy.array([[0], [0], [height]])

        x = numpy.linalg.solve(A.T.dot(A), A.T.dot(b))
        assert x[0][0] > 0
        assert x[1][0] > 0

        bottom_point = x[1][0] * bottom_ray_enu

        bottom_point = (bottom_point + left_right_position) / 2

        position1 = numpy.array([
            [bottom_point[0][0] + a_rot[0][0]],
            [bottom_point[1][0] + a_rot[1][0]],
            [bottom_point[2][0]],
        ])
        position2 = numpy.array([
            [bottom_point[0][0] + b_rot[0][0]],
            [bottom_point[1][0] + b_rot[1][0]],
            [bottom_point[2][0]],
        ])
        position3 = numpy.array([
            [bottom_point[0][0] + c_rot[0][0]],
            [bottom_point[1][0] + c_rot[1][0]],
            [bottom_point[2][0]],
        ])
        position4 = numpy.array([
            [bottom_point[0][0] + d_rot[0][0]],
            [bottom_point[1][0] + d_rot[1][0]],
            [bottom_point[2][0]],
        ])

        ecef1 = car_enu.dot(position1) + camera_position
        ecef2 = car_enu.dot(position2) + camera_position
        ecef3 = car_enu.dot(position3) + camera_position
        ecef4 = car_enu.dot(position4) + camera_position

        lla1 = pygeo.ECEFToLLA(ecef1.T).flatten()
        lla2 = pygeo.ECEFToLLA(ecef2.T).flatten()
        lla3 = pygeo.ECEFToLLA(ecef3.T).flatten()
        lla4 = pygeo.ECEFToLLA(ecef4.T).flatten()

        pglla1 = func.ST_SetSRID(
            func.ST_MakePoint(lla1[1], lla1[0], lla1[2]), 4326)
        pglla2 = func.ST_SetSRID(
            func.ST_MakePoint(lla2[1], lla2[0], lla2[2]), 4326)
        pglla3 = func.ST_SetSRID(
            func.ST_MakePoint(lla3[1], lla3[0], lla3[2]), 4326)
        pglla4 = func.ST_SetSRID(
            func.ST_MakePoint(lla4[1], lla4[0], lla4[2]), 4326)

        collected = func.ST_Collect(pglla1, pglla2)
        collected = func.ST_Collect(collected, pglla3)
        collected = func.ST_Collect(collected, pglla4)
        geom = func.ST_ConvexHull(collected)

        world = car_enu.dot(bottom_point) + camera_position
        lla = pygeo.ECEFToLLA(world.T).flatten()
        pglla = func.ST_SetSRID(
            func.ST_MakePoint(lla[1], lla[0], lla[2]), 4326)

        yield pglla, geom, vehicle_type, total_angle


@task
def detect(pid, model_filename):
    """Runs DPM and computes 3D pose."""

    logger = logging.getLogger('detect')
    logger.info((pid, model_filename))

    session = SESSION()
    try:
        # pylint: disable-msg=E1101
        num_detections, = session.query(func.count(Detection.id)) \
            .join(Model) \
            .filter(Detection.pid == pid) \
            .filter(Model.filename == model_filename) \
            .one()

        if num_detections > 0:
            logger.info('Already computed')
            return pid

        model = session.query(Model) \
            .filter_by(filename=model_filename) \
            .one()

        photo = session.query(Photo) \
            .options(joinedload('dataset')) \
            .filter_by(id=pid) \
            .one()

        vehicle_types = session.query(VehicleType) \
            .filter(VehicleType.id.in_([202, 8, 150, 63, 123, 16]))

        pydro_model = pydro.io.LoadModel(model.filename)
        image = scipy.misc.imread(
            os.path.join(IMAGE_DIR, photo.filename))
        pyramid = pydro.features.BuildPyramid(image, model=pydro_model)
        filtered_model = pydro_model.Filter(pyramid)
        parse_trees = list(filtered_model.Parse(model.thresh))

        # make sure we use at least one entry so we know we tried
        if len(parse_trees) == 0:
            parse_trees = list(
                itertools.islice(filtered_model.Parse(-numpy.inf), 1))

        assert len(parse_trees) > 0

        bbox_tuple = namedtuple('bbox_tuple', 'x1,x2,y1,y2')

        for tree in parse_trees:
            bbox = bbox_tuple(
                x1=tree.x1 / image.shape[1],
                x2=tree.x2 / image.shape[1],
                y1=tree.y1 / image.shape[0],
                y2=tree.y2 / image.shape[0],
            )
            score = tree.s
            angle = tree.child.rule.metadata.get('angle', None)

            if bbox.x1 > bbox.x2 or bbox.y1 > bbox.y2:
                continue

            car_pose_generator = compute_car_pose(
                photo,
                bbox,
                angle,
                vehicle_types
            )

            for lla, geom, vehicle_type, world_angle in car_pose_generator:
                det = Detection(
                    photo=photo,
                    x1=float(bbox.x1),
                    y1=float(bbox.y1),
                    x2=float(bbox.x2),
                    y2=float(bbox.y2),
                    score=float(score),
                    prob=float(
                        1.0 / (1.0 + math.exp(model.a * score + model.b))),
                    model=model,
                    angle=angle,
                    lla=lla,
                    geom=geom,
                    world_angle=float(world_angle),
                    vehicle_type=vehicle_type,
                )
                session.add(det)

        session.commit()

        return pid

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--pid', type=int, required=True)
    PARSER.add_argument('--model', required=True)
    ARGS = PARSER.parse_args()

    detect(
        pid=ARGS.pid,
        model_filename=ARGS.model,
    )
