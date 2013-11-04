"""A set of utility functions for querying the database."""

from sqlalchemy import func, desc
from nyc3dcars import Photo, Model, Vehicle, Detection
import math
import numpy


def overlap(geom_a, geom_b):
    """Computes the overlap between two bounding boxes."""

    intersection_score = intersection(geom_a, geom_b)
    area1 = (geom_a.x2 - geom_a.x1) * (geom_a.y2 - geom_a.y1)
    area2 = (geom_b.y2 - geom_b.y1) * (geom_b.x2 - geom_b.x1)
    union_score = area1 + area2 - intersection_score
    overlap_score = intersection_score / union_score

    return overlap_score


def overlap_asym(geom_a, geom_b):
    """Computes an asymmetric overlap between two bounding boxes."""

    intersection_score = intersection(geom_a, geom_b)
    area2 = (geom_b.y2 - geom_b.y1) * (geom_b.x2 - geom_b.x1)
    overlap_score = intersection_score / area2

    return overlap_score


def intersection(geom_a, geom_b):
    """Computes the interesction of two bounding boxes."""

    intersection_score = func.greatest(0,
                                       (func.least(geom_a.x2, geom_b.x2) -
                                        func.greatest(geom_a.x1, geom_b.x1))) * \
        func.greatest(0,
                      (func.least(geom_a.y2, geom_b.y2) -
                       func.greatest(geom_a.y1, geom_b.y1)))

    return intersection_score


def union(geom_a, geom_b):
    """Computes the union of two bounding boxes."""

    intersection_score = intersection(geom_a, geom_b)
    area1 = (geom_a.x2 - geom_a.x1) * (geom_a.y2 - geom_a.y1)
    area2 = (geom_b.y2 - geom_b.y1) * (geom_b.x2 - geom_b.x1)
    union_score = area1 + area2 - intersection_score

    return union_score


def match(labels):
    """Computes matching between detections and annotations."""

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
    """Selects all detections that satisfy query filters."""

    # pylint: disable-msg=E1101
    detections = session.query(
        score.label('score')) \
        .join(Photo) \
        .join(Model) \
        .filter(Model.filename == model) \
        .filter(Photo.test == True) \
        # pylint: enable-msg=E1101

    for query_filter in query_filters:
        detections = detections.filter(query_filter)

    return detections.all()


def precision_recall_threshold(labels, detections, threshold):
    """Computes precision and recall for particular threshold."""

    thresholded_labels = [
        label for label in labels if label.score >= threshold]
    thresholded_detections = [
        detection for detection in detections if detection.score >= threshold]

    num_detections = len(thresholded_detections)

    selected = match(thresholded_labels)

    return len(selected), num_detections, threshold


def get_num_vehicles(session, query_filters):
    """Gets the total number of annotations."""

    # pylint: disable-msg=E1101
    num_vehicles_query = session.query(
        func.count(Vehicle.id)) \
        .join(Photo) \
        .filter(Photo.test == True) \
        # pylint: enable-msg=E1101

    for query_filter in query_filters:
        num_vehicles_query = num_vehicles_query.filter(query_filter)

    num_vehicles, = num_vehicles_query.one()
    return num_vehicles


def orientation_sim_threshold(labels, detections, threshold):
    """Computes orientation similarity and recall for particular threshold."""

    thresholded_labels = [
        label for label in labels if label.score >= threshold]
    thresholded_detections = [
        detection for detection in detections if detection.score >= threshold]

    num_detections = len(thresholded_detections)

    selected = match(thresholded_labels)

    return sum(s.orientation_similarity for s in selected), len(selected), num_detections


def precision_recall(session, score, detection_filters, vehicle_filters, model):
    """Computes precision-recall curve."""

    num_vehicles = get_num_vehicles(session, vehicle_filters)

    overlap_score = overlap(Detection, Vehicle)

    # pylint: disable-msg=E1101
    labels = session.query(
        overlap_score.label('overlap'),
        Vehicle.id.label('vid'),
        Detection.id.label('did'),
        score.label('score')) \
        .select_from(Detection) \
        .join(Photo) \
        .join(Vehicle) \
        .join(Model) \
        .filter(Model.filename == model) \
        .filter(Photo.test == True) \
        .filter(overlap_score > 0.5)
    # pylint: enable-msg=E1101

    for query_filter in detection_filters:
        labels = labels.filter(query_filter)

    for query_filter in vehicle_filters:
        labels = labels.filter(query_filter)

    labels = labels.order_by(desc(overlap_score)).all()

    detections = get_detections(session, score, detection_filters, model)

    # pylint: disable-msg=E1101
    range_query = session.query(
        func.min(Detection.score),
        func.max(Detection.score)) \
        .join(Photo) \
        .join(Model) \
        .filter(Model.filename == model) \
        .filter(Photo.test == True)
    # pylint: enable-msg=E1101

    for query_filter in detection_filters:
        range_query = range_query.filter(query_filter)

    low, high = range_query.one()

    model = session.query(Model) \
        .filter_by(filename=model) \
        .one()

    thresholds_linear = [1 - i / 499.0 for i in xrange(500)]
    step = (high - low) / 500.0
    thresholds_sigmoid = [
        1.0 / (1.0 + math.exp(model.a * (step * i + low) + model.b))
        for i in xrange(500)
    ]

    thresholds = thresholds_linear + thresholds_sigmoid
    thresholds.sort(key=lambda k: -k)

    thresholded = [precision_recall_threshold(labels, detections, threshold)
                   for threshold in thresholds]

    return numpy.array([(
        float(tp) / num_detections if num_detections > 0 else 1,
        float(tp) / num_vehicles if num_vehicles > 0 else 1,
        threshold,
    ) for tp, num_detections, threshold in thresholded])


def orientation_similarity(session, score, detection_filters,
                           vehicle_filters, model):
    """Computes orientation similarity-recall curve."""

    num_vehicles = get_num_vehicles(session, vehicle_filters)

    overlap_score = overlap(Detection, Vehicle)

    # pylint: disable-msg=E1101
    labels = session.query(
        overlap_score.label('overlap'),
        Vehicle.id.label('vid'),
        Detection.id.label('did'),
        (-Vehicle.theta / 180 * math.pi + math.pi).label('gt'),
        (Detection.world_angle).label('d'),
        ((1 + func.cos(-Vehicle.theta / 180 * math.pi + math.pi - Detection.world_angle))
         / 2).label('orientation_similarity'),
        score.label('score')) \
        .select_from(Detection) \
        .join(Photo) \
        .join(Vehicle) \
        .join(Model) \
        .filter(Model.filename == model) \
        .filter(Photo.test == True) \
        .filter(overlap_score > 0.5)
    # pylint: enable-msg=E1101

    for query_filter in detection_filters:
        labels = labels.filter(query_filter)

    for query_filter in vehicle_filters:
        labels = labels.filter(query_filter)

    labels = labels.order_by(desc(overlap_score)).all()

    detections = get_detections(session, score, detection_filters, model)

    # pylint: disable-msg=E1101
    range_query = session.query(
        func.min(Detection.score),
        func.max(Detection.score)) \
        .join(Photo) \
        .join(Model) \
        .filter(Model.filename == model) \
        .filter(Photo.test == True)
    # pylint: enable-msg=E1101

    for query_filter in detection_filters:
        range_query = range_query.filter(query_filter)

    low, high = range_query.one()

    model = session.query(Model) \
        .filter_by(filename=model) \
        .one()

    thresholds_linear = [1 - i / 499.0 for i in xrange(500)]
    step = (high - low) / 500.0
    thresholds_sigmoid = [
        1.0 / (1.0 + math.exp(model.a * (step * i + low) + model.b))
        for i in xrange(500)
    ]

    thresholds = thresholds_linear + thresholds_sigmoid
    thresholds.sort(key=lambda k: -k)

    thresholded = [orientation_sim_threshold(
        labels, detections, threshold) for threshold in thresholds]

    return numpy.array([(
        aos / num_detections if num_detections > 0 else 1,
        float(tp) / num_vehicles if num_vehicles > 0 else 1
    ) for aos, tp, num_detections in thresholded])
