#!/usr/bin/env python

"""Compute horizontal and vertical 3D localization errors."""

from query_utils import precision_recall, overlap

import scores
import logging
import numpy
from nyc3dcars import SESSION, Detection, Vehicle, Photo, Model
import argparse
from sqlalchemy import func, desc, and_


def get_labels(session, score, detection_filters, vehicle_filters,
               model, threshold):
    """Retrieves all possible detection-annotation pairings
       that satify the VOC criterion."""

    overlap_score = overlap(Detection, Vehicle)

    # pylint: disable-msg=E1101
    dist_x = (func.ST_X(func.ST_Transform(Detection.lla, 102718))
              - func.ST_X(func.ST_Transform(Vehicle.lla, 102718))) \
        * 0.3048
    dist_y = (func.ST_Y(func.ST_Transform(Detection.lla, 102718))
              - func.ST_Y(func.ST_Transform(Vehicle.lla, 102718))) \
        * 0.3048
    dist = func.sqrt(dist_x * dist_x + dist_y * dist_y)
    height_diff = func.abs(
        func.ST_Z(Detection.lla) - func.ST_Z(Vehicle.lla))

    labels = session.query(
        overlap_score.label('overlap'),
        Vehicle.id.label('vid'),
        Detection.id.label('did'),
        dist.label('dist'),
        height_diff.label('height_diff'),
        score.label('score')) \
        .select_from(Detection) \
        .join(Photo) \
        .join(Vehicle) \
        .join(Model) \
        .filter(Model.filename == model) \
        .filter(Photo.test == True) \
        .filter(overlap_score > 0.5) \
        .filter(score > threshold)
    # pylint: enable-msg=E1101

    for query_filter in detection_filters:
        labels = labels.filter(query_filter)

    for query_filter in vehicle_filters:
        labels = labels.filter(query_filter)

    labels = labels.order_by(desc(overlap_score)).all()

    return labels


def gen_dist_error(model, methods, dataset_id):
    """Generates the horizontal and vertical 3D error statistics."""

    session = SESSION()
    try:
        # pylint: disable-msg=E1101
        model_id, = session.query(Model.id) \
            .filter_by(filename=model) \
            .one()

        todo, = session.query(func.count(Photo.id)) \
            .outerjoin((
                Detection,
                and_(
                    Detection.pid == Photo.id,
                    Detection.mid == model_id
                )
            )) \
            .filter(Photo.test == True) \
            .filter(Detection.id == None) \
            .filter(Photo.dataset_id == dataset_id) \
            .one()
        # pylint: enable-msg=E1101

        if todo > 0:
            msg = '%s is not ready. %d photos remaining' % (model, todo)
            logging.info(msg)
            return

        not_ready = False

        for name in methods:
            nms_method = scores.METHODS[name]
            # pylint: disable-msg=E1101
            todo, = session.query(func.count(Detection.id)) \
                .join(Model) \
                .join(Photo) \
                .filter(Photo.test == True) \
                .filter(Model.filename == model) \
                .filter(Photo.dataset_id == dataset_id) \
                .filter(nms_method.output == None) \
                .one()
            # pylint: enable-msg=E1101
            if todo > 0:
                msg = '%s is not ready.  %d %s NMS remaining' % (
                    model, todo, name)
                logging.info(msg)
                not_ready = True

        if not_ready:
            return

        # pylint: disable-msg=E1101
        dataset_id = [Photo.dataset_id == dataset_id]
        # pylint: enable-msg=E1101

        for method in methods:
            nms_method = scores.METHODS[method]
            selected = [nms_method.output == True]
            msg = '%s method: %s' % (model, method)
            logging.info(msg)
            points = precision_recall(
                session,
                nms_method.score,
                dataset_id + selected,
                dataset_id, model
            )
            idx = numpy.abs(points[:, 0] - 0.9).argmin()
            logging.info(points[idx, :])
            labels = get_labels(
                session, nms_method.score, selected, [], model, points[idx, 2])
            dists = [l.dist for l in labels]
            dist_zs = [l.height_diff for l in labels]
            logging.info(
                (numpy.array(dists).mean(), numpy.array(dist_zs).mean()))

        logging.info('done')

    except:
        raise
    finally:
        session.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--model', required=True)
    PARSER.add_argument('--dataset-id', required=True, type=int)
    PARSER.add_argument('--methods', nargs='+',
                        default=['reference', 'coverage',
                                 'horizon', 'height2',
                                 'angle2', 'all2']
                        )
    ARGS = PARSER.parse_args()

    gen_dist_error(
        model=ARGS.model,
        methods=ARGS.methods,
        dataset_id=ARGS.dataset_id,
    )
