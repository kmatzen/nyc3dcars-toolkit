#!/usr/bin/env python

import query_utils

import scores

import logging
import itertools
import math
import scipy.integrate
import numpy
import nyc3dcars
import argparse
from sqlalchemy import desc, func, and_


def match(labels):
    true_positive = []
    covered = []
    selected = []

    for label in labels:
        if label.did in true_positive:
            continue
        if label.vid in covered:
            continue
        true_positive += [label.did]
        covered += [label.vid]
        selected += [label]
    return selected


def get_detections(session, score, query_filters, model):
    detections = session.query(
        score.label('score')) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \

    for query_filter in query_filters:
        detections = detections.filter(query_filter)

    return detections.all()


def precision_recall_threshold(labels, detections, threshold):
    thresholded_labels = [
        label for label in labels if label.score >= threshold]
    thresholded_detections = [
        detection for detection in detections if detection.score >= threshold]

    num_detections = len(thresholded_detections)

    selected = match(thresholded_labels)

    return len(selected), num_detections


def orientation_similarity_threshold(labels, detections, threshold):
    thresholded_labels = [
        label for label in labels if label.score >= threshold]
    thresholded_detections = [
        detection for detection in detections if detection.score >= threshold]

    num_detections = len(thresholded_detections)

    selected = match(thresholded_labels)

    return sum(s.orientation_similarity for s in selected), len(selected), num_detections


def get_num_vehicles(session, query_filters):
    num_vehicles_query = session.query(
        func.count(nyc3dcars.Vehicle.id)) \
        .join(nyc3dcars.Photo) \
        .filter(nyc3dcars.Photo.test == True) \

    for query_filter in query_filters:
        num_vehicles_query = num_vehicles_query.filter(query_filter)

    num_vehicles, = num_vehicles_query.one()
    return num_vehicles


def precision_recall(session, score, detection_filters, vehicle_filters, model):
    num_vehicles = get_num_vehicles(session, vehicle_filters)

    overlap_score = query_utils.overlap(nyc3dcars.Detection, nyc3dcars.Vehicle)
    labels = session.query(
        overlap_score.label('overlap'),
        nyc3dcars.Vehicle.id.label('vid'),
        nyc3dcars.Detection.id.label('did'),
        score.label('score')) \
        .select_from(nyc3dcars.Detection) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Vehicle) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \
        .filter(overlap_score > 0.5)

    for query_filter in detection_filters:
        labels = labels.filter(query_filter)

    for query_filter in vehicle_filters:
        labels = labels.filter(query_filter)

    labels = labels.order_by(desc(overlap_score)).all()

    detections = get_detections(session, score, detection_filters, model)

    range_query = session.query(
        func.min(nyc3dcars.Detection.score),
        func.max(nyc3dcars.Detection.score)) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \

    for query_filter in detection_filters:
        range_query = range_query.filter(query_filter)

    low, high = range_query.one()

    model = session.query(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .one()

    thresholds_linear = [1 - i / 499.0 for i in xrange(500)]
    step = (high - low) / 500.0
    thresholds_sigmoid = [1.0 / (1.0 + math.exp(model.a * (step * i + low) + model.b))
                          for i in xrange(500)]

    thresholds = thresholds_linear + thresholds_sigmoid
    thresholds.sort(key=lambda k: -k)

    thresholded = [precision_recall_threshold(labels, detections, threshold)
                   for threshold in thresholds]

    return numpy.array([(float(tp) / num_detections if num_detections > 0 else 1, float(tp) / num_vehicles if num_vehicles > 0 else 1) for tp, num_detections in thresholded])


def orientation_similarity(session, score, detection_filters, vehicle_filters, model):
    num_vehicles = get_num_vehicles(session, vehicle_filters)

    overlap_score = query_utils.overlap(nyc3dcars.Detection, nyc3dcars.Vehicle)

    labels = session.query(
        overlap_score.label('overlap'),
        nyc3dcars.Vehicle.id.label('vid'),
        nyc3dcars.Detection.id.label('did'),
        (-nyc3dcars.Vehicle.theta / 180 * math.pi + math.pi).label('gt'),
        (nyc3dcars.Detection.world_angle).label('d'),
        ((1 + func.cos(-nyc3dcars.Vehicle.theta / 180 * math.pi + math.pi - nyc3dcars.Detection.world_angle))
         / 2).label('orientation_similarity'),
        score.label('score')) \
        .select_from(nyc3dcars.Detection) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Vehicle) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \
        .filter(overlap_score > 0.5)

    for query_filter in detection_filters:
        labels = labels.filter(query_filter)

    for query_filter in vehicle_filters:
        labels = labels.filter(query_filter)

    labels = labels.order_by(desc(overlap_score)).all()

    detections = get_detections(session, score, detection_filters, model)

    range_query = session.query(
        func.min(nyc3dcars.Detection.score),
        func.max(nyc3dcars.Detection.score)) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \

    for query_filter in detection_filters:
        range_query = range_query.filter(query_filter)

    low, high = range_query.one()

    model = session.query(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .one()

    thresholds_linear = [1 - i / 499.0 for i in xrange(500)]
    step = (high - low) / 500.0
    thresholds_sigmoid = [1.0 / (1.0 + math.exp(model.a * (step * i + low) + model.b))
                          for i in xrange(500)]

    thresholds = thresholds_linear + thresholds_sigmoid
    thresholds.sort(key=lambda k: -k)

    thresholded = [orientation_similarity_threshold(
        labels, detections, threshold) for threshold in thresholds]

    return numpy.array([(aos / num_detections if num_detections > 0 else 1, float(tp) / num_vehicles if num_vehicles > 0 else 1) for aos, tp, num_detections in thresholded])


def gen_results(model, methods, aos, dataset_id):
    try:
        session = nyc3dcars.SESSION()

        difficulties = {
            'full': []
        }

        daynights = {
            'both': [],
        }

        model_id, = session.query(nyc3dcars.Model.id) \
            .filter(nyc3dcars.Model.filename == model) \
            .one()

        todo, = session.query(func.count(nyc3dcars.Photo.id)) \
            .outerjoin((nyc3dcars.Detection, and_(nyc3dcars.Detection.pid == nyc3dcars.Photo.id, nyc3dcars.Detection.pmid == model_id))) \
            .filter(nyc3dcars.Photo.test == True) \
            .filter(nyc3dcars.Detection.id == None) \
            .filter(nyc3dcars.Photo.dataset_id == dataset_id) \
            .one()

        if todo > 0:
            logging.info(
                '%s is not ready. %d photos remaining' % (model, todo))
            return

        not_ready = False

        for name in methods:
            nms_method = scores.METHODS[name]
            todo, = session.query(func.count(nyc3dcars.Detection.id)) \
                .join(nyc3dcars.Model) \
                .join(nyc3dcars.Photo) \
                .filter(nyc3dcars.Photo.test == True) \
                .filter(nyc3dcars.Model.filename == model) \
                .filter(nyc3dcars.Photo.dataset_id == dataset_id) \
                .filter(nms_method.output == None) \
                .one()
            if todo > 0:
                logging.info(
                    '%s is not ready.  %d %s NMS remaining' % (model, todo, name))
                not_ready = True

        if not_ready:
            return

        dataset_id = [nyc3dcars.Photo.dataset_id == dataset_id]

        for daynight, difficulty in itertools.product(daynights, difficulties):
            for method in methods:
                nms_method = scores.METHODS[method]
                selected = [nms_method.output == True]
                logging.info('%s daynight: %s, difficulty: %s, method: %s' %
                             (model, daynight, difficulty, method))
                points = precision_recall(session, nms_method.score, dataset_id + daynights[
                                          daynight] + selected, dataset_id + daynights[daynight] + difficulties[difficulty], model)
                name = '%s, %s' % (model, method)
                if aos:
                    points_aos = orientation_similarity(session, nms_method.score, dataset_id + daynights[
                                                        daynight] + selected, dataset_id + daynights[daynight] + difficulties[difficulty], model)
                else:
                    points_aos = None

                print(points.shape)
                print(scipy.integrate.trapz(points[:, 0], points[:, 1]))
                filename = '%s-%s-%s-%s-pr.txt' % (
                    model, daynight, difficulty, method)
                print(filename)
                numpy.savetxt(filename, points)
                if points_aos is not None:
                    print(scipy.integrate.trapz(
                        points_aos[:, 0], points_aos[:, 1]))
                    filename = '%s-%s-%s-%s-aos.txt' % (
                        model, daynight, difficulty, method)
                    print(filename)
                    numpy.savetxt(filename, points_aos)

        logging.info('done')

    except:
        raise
    finally:
        session.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--aos', action='store_true')
    parser.add_argument('--dataset-id', required=True, type=int)
    parser.add_argument('--methods', nargs='+', default=scores.METHODS.keys())
    args = parser.parse_args()

    gen_results(**vars(args))
