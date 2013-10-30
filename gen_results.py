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
from sqlalchemy import *

from pylab import *

def match(labels):
    true_positive = []
    covered = []
    selected = []

    for t in labels:
        if t.did in true_positive:
            continue
        if t.vid in covered:
            continue
        true_positive += [t.did]
        covered += [t.vid]
        selected += [t]
    return selected

def gen_angle_hist(labels, threshold):
    selected = match(labels, threshold)

    return numpy.array([label.angle_diff if label.angle_diff < math.pi else label.angle_diff - 2*math.pi for label in selected])

def gen_dist_hist(labels, threshold):
    selected = match(labels, threshold)

    dist_x = [label.dist_x for label in selected]
    dist_y = [label.dist_y for label in selected]

    return dist_x, dist_y

def gen_height_hist(labels, threshold):
    selected = match(labels, threshold)

    return numpy.array([label.height_diff for label in selected])

def get_detections(session, score, filters, model):
    detections = session.query(
        score.label('score')) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \

    for filter in filters:
        detections = detections.filter(filter)

    return detections.all()

def precision_recall_threshold(session, labels, detections, threshold):
    l = [label for label in labels if label.score >= threshold]
    d = [detection for detection in detections if detection.score >= threshold]

    nDetections = len(d)

    selected = match(l)

    return len(selected), nDetections

def orientation_similarity_threshold(session, labels, detections, threshold):
    l = [label for label in labels if label.score >= threshold]
    d = [detection for detection in detections if detection.score >= threshold]

    nDetections = len(d)

    selected = match(l)

    return sum(s.orientation_similarity for s in selected), len(selected), nDetections


def get_nVehicles(session, filters):
    nVehicles_q = session.query(
        func.count(nyc3dcars.Vehicle.id)) \
        .join(nyc3dcars.Photo) \
        .filter(nyc3dcars.Photo.test == True) \

    for filter in filters:
        nVehicles_q = nVehicles_q.filter(filter)

    nVehicles, = nVehicles_q.one()
    return nVehicles

def precision_recall(session, score, detection_filters, vehicle_filters, model, voc_threshold):
    nVehicles = get_nVehicles(session, vehicle_filters)

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
        .filter(overlap_score > 0.5)#voc_threshold) 

    for filter in detection_filters:
        labels = labels.filter(filter)

    for filter in vehicle_filters:
        labels = labels.filter(filter)

    labels = labels.order_by(desc(overlap_score)).all()

    detections = get_detections(session, score, detection_filters, model)

    range_query = session.query(
        func.min(nyc3dcars.Detection.score),
        func.max(nyc3dcars.Detection.score)) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \
        
    for filter in detection_filters:
        range_query = range_query.filter(filter)

    low, high = range_query.one()

    model = session.query(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .one()

    thresholds_linear = [1-i/499.0 for i in xrange(500)]
    step = (high - low)/500.0
    thresholds_sigmoid = [1.0/(1.0 + math.exp(model.a*(step*i + low) + model.b)) for i in xrange(500)]

    thresholds = thresholds_linear + thresholds_sigmoid
    thresholds.sort(key=lambda k: -k)

    thresholded = [precision_recall_threshold(session, labels, detections, threshold) for threshold in thresholds]

    return numpy.array([(float(tp)/nDetections if nDetections > 0 else 1, float(tp)/nVehicles if nVehicles > 0 else 1) for tp, nDetections in thresholded])

def orientation_similarity(session, score, detection_filters, vehicle_filters, model, voc_threshold):
    nVehicles = get_nVehicles(session, vehicle_filters)

    overlap_score = query_utils.overlap(nyc3dcars.Detection, nyc3dcars.Vehicle)

    labels = session.query(
        overlap_score.label('overlap'),
        nyc3dcars.Vehicle.id.label('vid'),
        nyc3dcars.Detection.id.label('did'),
        (-nyc3dcars.Vehicle.theta/180*math.pi+math.pi).label('gt'),
        (nyc3dcars.Detection.world_angle).label('d'),
        ((1 + func.cos(-nyc3dcars.Vehicle.theta/180*math.pi+math.pi-nyc3dcars.Detection.world_angle))/2).label('orientation_similarity'),
        score.label('score')) \
        .select_from(nyc3dcars.Detection) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Vehicle) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \
        .filter(overlap_score > 0.5)#voc_threshold) 

    for filter in detection_filters:
        labels = labels.filter(filter)

    for filter in vehicle_filters:
        labels = labels.filter(filter)

    labels = labels.order_by(desc(overlap_score)).all()

    detections = get_detections(session, score, detection_filters, model)

    range_query = session.query(
        func.min(nyc3dcars.Detection.score),
        func.max(nyc3dcars.Detection.score)) \
        .join(nyc3dcars.Photo) \
        .join(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .filter(nyc3dcars.Photo.test == True) \
        
    for filter in detection_filters:
        range_query = range_query.filter(filter)

    low, high = range_query.one()

    model = session.query(nyc3dcars.Model) \
        .filter(nyc3dcars.Model.filename == model) \
        .one()

    thresholds_linear = [1-i/499.0 for i in xrange(500)]
    step = (high - low)/500.0
    thresholds_sigmoid = [1.0/(1.0 + math.exp(model.a*(step*i + low) + model.b)) for i in xrange(500)]

    thresholds = thresholds_linear + thresholds_sigmoid
    thresholds.sort(key=lambda k: -k)

    thresholded = [orientation_similarity_threshold(session, labels, detections, threshold) for threshold in thresholds]

    return numpy.array([(aos/nDetections if nDetections > 0 else 1, float(tp)/nVehicles if nVehicles > 0 else 1) for aos, tp, nDetections in thresholded])


def gen_results(model, methods, aos, clobber, dataset_id):
    try:
        session = nyc3dcars.Session()

        difficulties = {
            'full':[]
        }

        daynights = {
            'both':[],
        }

        model_id, = session.query(nyc3dcars.Model.id) \
            .filter(nyc3dcars.Model.filename == model) \
            .one()

        todo, = session.query(func.count(nyc3dcars.Photo.id)) \
            .outerjoin((nyc3dcars.Detection,and_(nyc3dcars.Detection.pid == nyc3dcars.Photo.id, nyc3dcars.Detection.pmid == model_id))) \
            .filter(nyc3dcars.Photo.test == True) \
            .filter(nyc3dcars.Detection.id == None) \
            .filter(nyc3dcars.Photo.dataset_id == dataset_id) \
            .one()

        if todo > 0:
            logging.info('%s is not ready. %d photos remaining'%(model, todo))
            return

        not_ready = False

        for name in methods: 
            nms_method = scores.Methods[name]
            todo, = session.query(func.count(nyc3dcars.Detection.id)) \
                .join(nyc3dcars.Model) \
                .join(nyc3dcars.Photo) \
                .filter(nyc3dcars.Photo.test == True) \
                .filter(nyc3dcars.Model.filename == model) \
                .filter(nyc3dcars.Photo.dataset_id == dataset_id) \
                .filter(nms_method.output == None) \
                .one()
            if todo > 0:
                logging.info('%s is not ready.  %d %s NMS remaining'%(model, todo, name))
                not_ready = True

        if not_ready:
            return
 
        voc_threshold = 0.5

        dataset_id = [nyc3dcars.Photo.dataset_id == dataset_id]

        for daynight, difficulty in itertools.product(daynights, difficulties):
            for method in methods:
                nms_method = scores.Methods[method]
                selected = [nms_method.output == True]
                logging.info('%s daynight: %s, difficulty: %s, method: %s'%(model, daynight, difficulty, method))
                points = precision_recall(session, nms_method.score, dataset_id +daynights[daynight]+selected, dataset_id+daynights[daynight] + difficulties[difficulty], model, voc_threshold)
                name = '%s, %s'%(model,method)
                if aos:
                    points_aos = orientation_similarity(session, nms_method.score, dataset_id+daynights[daynight]+selected, dataset_id+daynights[daynight] + difficulties[difficulty], model, voc_threshold)
                else:
                    points_aos = None

                print(points.shape)
                print(scipy.integrate.trapz(points[:,0], points[:,1]))
                filename = '%s-%s-%s-%s-pr.txt'%(model,daynight,difficulty,method)
                print(filename)
                numpy.savetxt(filename, points)
                if points_aos is not None:
                    print(scipy.integrate.trapz(points_aos[:,0], points_aos[:,1]))
                    filename = '%s-%s-%s-%s-aos.txt'%(model,daynight,difficulty,method)
                    print(filename)
                    numpy.savetxt(filename, points_aos)
                #plot(points[:,1], points[:,0])
                #show()

        
        session.commit() 
 
        logging.info('done') 

    except:
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--aos', action='store_true')
    parser.add_argument('--clobber', action='store_true')
    parser.add_argument('--dataset-id', required=True, type=int)
    parser.add_argument('--methods', nargs='+', default=scores.Methods.keys())
    args = parser.parse_args()

    gen_results(**vars(args))
