#!/usr/bin/env python

"""Module performs the processing from Labeler to NYC3DCars dataset."""

import numpy
import math
from nyc3dcars import SESSION, Photo, VehicleType, Vehicle
import labeler
from sqlalchemy import func
from sqlalchemy.orm import joinedload
import pygeo


def convert_vehicle(nyc3dcars_session, labeler_vehicle):
    """Converts the vehicle from Labeler format to NYC3DCars format."""

    photo, lat, lon, alt = nyc3dcars_session.query(
        Photo,
        func.ST_Y(Photo.lla),
        func.ST_X(Photo.lla),
        func.ST_Z(Photo.lla)) \
        .filter_by(id=labeler_vehicle.revision.annotation.pid) \
        .options(joinedload('dataset')) \
        .one()
    left = labeler_vehicle.x1
    right = labeler_vehicle.x2
    top = labeler_vehicle.y1
    bottom = labeler_vehicle.y2

    camera_lla = numpy.array([[lat], [lon], [alt]])
    camera_enu = pygeo.LLAToENU(camera_lla.T).reshape((3, 3))
    dataset_correction = numpy.array([
        [photo.dataset.t1],
        [photo.dataset.t2],
        [photo.dataset.t3],
    ])
    camera_rotation = numpy.array([
        [photo.r11, photo.r12, photo.r13],
        [photo.r21, photo.r22, photo.r23],
        [photo.r31, photo.r32, photo.r33],
    ])

    camera_up = camera_enu.T.dot(
        camera_rotation.T.dot(numpy.array([[0], [1], [0]])))
    offset = numpy.array([[-labeler_vehicle.x], [-labeler_vehicle.z], [0]])
    camera_offset = camera_up * \
        labeler_vehicle.revision.cameraheight / camera_up[2]
    total_offset = offset - camera_offset
    ecef_camera = pygeo.LLAToECEF(camera_lla.T).T
    ecef_camera += dataset_correction
    ecef_total_offset = camera_enu.dot(total_offset)
    vehicle_ecef = ecef_camera + ecef_total_offset

    vehicle_type = labeler_vehicle.type
    model = nyc3dcars_session.query(VehicleType) \
        .filter_by(label=vehicle_type) \
        .one()

    vehicle_lla = pygeo.ECEFToLLA(vehicle_ecef.T).T

    theta = math.radians(-labeler_vehicle.theta)
    mlength = model.length
    mwidth = model.width
    car_a = -math.sin(theta) * 0.3048 * \
        mlength / 2 + math.cos(theta) * 0.3048 * mwidth / 2
    car_b = math.cos(theta) * 0.3048 * mlength / \
        2 + math.sin(theta) * 0.3048 * mwidth / 2
    car_c = math.sin(theta) * 0.3048 * mlength / \
        2 + math.cos(theta) * 0.3048 * mwidth / 2
    car_d = -math.cos(theta) * 0.3048 * \
        mlength / 2 + math.sin(theta) * 0.3048 * mwidth / 2
    car_corner_offset1 = camera_enu.dot(numpy.array([[car_a], [car_b], [0]]))
    car_corner_offset2 = camera_enu.dot(numpy.array([[car_c], [car_d], [0]]))

    car_corner1 = pygeo.ECEFToLLA(
        (vehicle_ecef + car_corner_offset1).T).T.flatten()
    car_corner2 = pygeo.ECEFToLLA(
        (vehicle_ecef - car_corner_offset1).T).T.flatten()
    car_corner3 = pygeo.ECEFToLLA(
        (vehicle_ecef + car_corner_offset2).T).T.flatten()
    car_corner4 = pygeo.ECEFToLLA(
        (vehicle_ecef - car_corner_offset2).T).T.flatten()

    pg_corner1 = func.ST_SetSRID(
        func.ST_MakePoint(car_corner1[1], car_corner1[0], car_corner1[2]), 4326)
    pg_corner2 = func.ST_SetSRID(
        func.ST_MakePoint(car_corner2[1], car_corner2[0], car_corner2[2]), 4326)
    pg_corner3 = func.ST_SetSRID(
        func.ST_MakePoint(car_corner3[1], car_corner3[0], car_corner3[2]), 4326)
    pg_corner4 = func.ST_SetSRID(
        func.ST_MakePoint(car_corner4[1], car_corner4[0], car_corner4[2]), 4326)

    collection = func.ST_Collect(pg_corner1, pg_corner2)
    collection = func.ST_Collect(collection, pg_corner3)
    collection = func.ST_Collect(collection, pg_corner4)

    car_polygon = func.ST_ConvexHull(collection)

    camera_ecef = pygeo.LLAToECEF(camera_lla.T).T
    vehicle_ecef = pygeo.LLAToECEF(vehicle_lla.T).T

    diff = camera_ecef - vehicle_ecef

    normalized = diff / numpy.linalg.norm(diff)

    vehicle_enu = pygeo.LLAToENU(vehicle_lla.T).reshape((3, 3))

    rotated = vehicle_enu.T.dot(normalized)

    theta = func.acos(rotated[2][0])

    view_phi = func.atan2(rotated[1][0], rotated[0][0])

    vehicle_phi = math.radians(-labeler_vehicle.theta)

    phi = vehicle_phi - view_phi

    out = nyc3dcars_session.query(
        theta.label('theta'),
        phi.label('phi')) \
        .one()
    out.phi = ((out.phi + math.pi) % (2 * math.pi)) - math.pi
    out.theta = ((out.theta + math.pi) % (2 * math.pi)) - math.pi
    view_phi = out.phi
    view_theta = out.theta

    left = labeler_vehicle.x1
    right = labeler_vehicle.x2
    top = labeler_vehicle.y1
    bottom = labeler_vehicle.y2

    for bbox_session in labeler_vehicle.bbox_sessions:
        if not bbox_session.user.trust:
            continue

        print((
            bbox_session.user.username,
            labeler_vehicle.revision.annotation.pid
        ))
        left = bbox_session.x1
        right = bbox_session.x2
        top = bbox_session.y1
        bottom = bbox_session.y2
        break

    occlusions = [
        occlusion.category for occlusion in labeler_vehicle.occlusionrankings
        if occlusion.occlusion_session.user.trust and occlusion.category != 5
    ]

    if len(occlusions) == 0:
        return

    pg_lla = func.ST_SetSRID(
        func.ST_MakePoint(vehicle_lla[1][0], vehicle_lla[0][0], vehicle_lla[2][0]), 4326)

    nyc3dcars_vehicle = Vehicle(
        id=labeler_vehicle.id,
        pid=photo.id,
        x=labeler_vehicle.x,
        z=labeler_vehicle.z,
        theta=labeler_vehicle.theta,
        x1=left,
        x2=right,
        y1=top,
        y2=bottom,
        type_id=model.id,
        occlusion=min(occlusions),
        geom=car_polygon,
        lla=pg_lla,
        view_theta=view_theta,
        view_phi=view_phi,
        cropped=labeler_vehicle.cropped,
    )
    nyc3dcars_session.add(nyc3dcars_vehicle)


def select_evaluation():
    """Selects the vehicles that will appear in NYC3DCars."""

    nyc3dcars_session = SESSION()
    labeler_session = labeler.SESSION()

    try:
        # Reset photos
        photos = nyc3dcars_session.query(Photo) \
            .options(joinedload('dataset'))

        for photo in photos:
            photo.daynight = None

        # Reset vehicles
        vehicles = nyc3dcars_session.query(Vehicle)
        for vehicle in vehicles:
            nyc3dcars_session.delete(vehicle)

        # Turn photos back on if they have at least 1 final revision
        # pylint: disable-msg=E1101
        photos = labeler_session.query(labeler.Photo) \
            .select_from(labeler.Photo) \
            .join(labeler.Annotation) \
            .join(labeler.User) \
            .join(labeler.Revision) \
            .options(joinedload('daynights')) \
            .options(joinedload('annotations.flags')) \
            .options(joinedload('annotations')) \
            .filter(labeler.Revision.final == True) \
            .filter(labeler.User.trust == True) \
            .distinct()
        # pylint: enable-msg=E1101

        photos = list(photos)

        num_photos = len(photos)

        num_test = 0
        num_train = 0
        num_flagged = 0

        print('Checking for new photos')
        for labeler_photo in photos:
            # Do not consider photos that have been flagged
            num_flags = sum(
                len(annotation.flags)
                for annotation in labeler_photo.annotations
                if annotation.user.trust
            )
            if num_flags > 0:
                num_flagged += 1
                continue

            days = 0
            nights = 0

            for daynight in labeler_photo.daynights:
                if not daynight.user.trust:
                    continue
                if daynight.daynight == 'day':
                    days += 1
                else:
                    nights += 1
            if days + nights == 0:
                print('Need Day/Night for photo: %d' % labeler_photo.id)
                continue

            nyc3dcars_photo = nyc3dcars_session.query(Photo) \
                .filter_by(id=labeler_photo.id) \
                .one()

            if nyc3dcars_photo.test == True:
                num_test += 1
            elif nyc3dcars_photo.test == False:
                num_train += 1
            else:
                if num_train > num_test:
                    print('Test: %d' % labeler_photo.id)
                    nyc3dcars_photo.test = True
                    num_test += 1
                else:
                    print('Train: %d' % labeler_photo.id)
                    nyc3dcars_photo.test = False
                    num_train += 1

            if days > nights:
                nyc3dcars_photo.daynight = 'day'
            else:
                nyc3dcars_photo.daynight = 'night'
        print('New photos done')
        print('%d photos' % num_photos)
        print('%d flagged' % num_flagged)
        print('%d test' % num_test)
        print('%d train' % num_train)

        # pylint: disable-msg=E1101
        good_pids = nyc3dcars_session.query(Photo.id) \
            .filter(Photo.test != None) \
            .all()
        # pylint: enable-msg=E1101

        # get photos with 1 and 2 users
        # pylint: disable-msg=E1101
        photos_one_user = labeler_session.query(labeler.Photo.id) \
            .select_from(labeler.Photo) \
            .join(labeler.Annotation) \
            .join(labeler.User) \
            .join(labeler.Revision) \
            .filter(labeler.User.trust == True) \
            .filter(labeler.Revision.final == True) \
            .filter(labeler.Photo.id.in_(good_pids)) \
            .group_by(labeler.Photo.id) \
            .having(func.count(labeler.Revision.id) == 1)

        photos_two_user = labeler_session.query(labeler.Photo.id) \
            .select_from(labeler.Photo) \
            .join(labeler.Annotation) \
            .join(labeler.User) \
            .join(labeler.Revision) \
            .filter(labeler.User.trust == True) \
            .filter(labeler.Revision.final == True) \
            .filter(labeler.Photo.id.in_(good_pids)) \
            .group_by(labeler.Photo.id) \
            .having(func.count(labeler.Revision.id) == 2)

        photos_more_user = labeler_session.query(labeler.Photo.id) \
            .select_from(labeler.Photo) \
            .join(labeler.Annotation) \
            .join(labeler.User) \
            .join(labeler.Revision) \
            .filter(labeler.User.trust == True) \
            .filter(labeler.Revision.final == True) \
            .filter(labeler.Photo.id.in_(good_pids)) \
            .group_by(labeler.Photo.id) \
            .having(func.count(labeler.Revision.id) > 2)
        for photo in photos_more_user:
            print(photo.id)

        for photo, in photos_one_user:
            vehicles = labeler_session.query(labeler.Vehicle) \
                .select_from(labeler.Vehicle) \
                .join(labeler.Revision) \
                .join(labeler.Annotation) \
                .join(labeler.User) \
                .options(joinedload('revision')) \
                .options(joinedload('revision.annotation')) \
                .options(joinedload('revision.annotation.user')) \
                .options(joinedload('occlusionrankings')) \
                .options(joinedload('occlusionrankings.occlusion_session')) \
                .options(joinedload('bbox_sessions')) \
                .filter(labeler.User.trust == True) \
                .filter(labeler.Annotation.pid == photo) \
                .filter(labeler.Revision.final == True) \
                .distinct()

            for vehicle in vehicles:
                convert_vehicle(nyc3dcars_session, vehicle)

        # get good vehicles for 2 user case
        for photo, in photos_two_user:
            annotations = labeler_session.query(labeler.Annotation) \
                .join(labeler.User) \
                .join(labeler.Revision) \
                .filter(labeler.User.trust == True) \
                .filter(labeler.Annotation.pid == photo) \
                .filter(labeler.Revision.final == True) \
                .all()
            assert len(annotations) == 2

            vehicles1 = labeler_session.query(labeler.Vehicle) \
                .join(labeler.Revision) \
                .join(labeler.Annotation) \
                .join(labeler.User) \
                .filter(labeler.User.trust == True) \
                .filter(labeler.Revision.aid == annotations[0].id) \
                .filter(labeler.Revision.final == True) \
                .all()

            vehicles2 = labeler_session.query(labeler.Vehicle) \
                .join(labeler.Revision) \
                .join(labeler.Annotation) \
                .join(labeler.User) \
                .filter(labeler.User.trust == True) \
                .filter(labeler.Revision.aid == annotations[1].id) \
                .filter(labeler.Revision.final == True) \
                .all()

            if len(vehicles1) > len(vehicles2):
                vehicles = vehicles1
            else:
                vehicles = vehicles2

            for vehicle in vehicles:
                print(vehicle.id)
                convert_vehicle(nyc3dcars_session, vehicle)

        num_vehicles, = nyc3dcars_session.query(
            func.count(Vehicle.id)) \
            .one()

        photo_test, = nyc3dcars_session.query(
            func.count(Photo.id)) \
            .filter(Photo.test == True) \
            .one()

        photo_train, = nyc3dcars_session.query(
            func.count(Photo.id)) \
            .filter(Photo.test == False) \
            .one()
        # pylint: enable-msg=E1101

        print('%d vehicles in dataset' % num_vehicles)
        print('%d images for training' % photo_train)
        print('%d images for testing' % photo_test)

        nyc3dcars_session.commit()
    except:
        nyc3dcars_session.rollback()
        labeler_session.rollback()
        raise
    finally:
        nyc3dcars_session.close()
        labeler_session.close()

if __name__ == '__main__':
    select_evaluation()
