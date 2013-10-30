from sqlalchemy import func


def overlap(geom_a, geom_b):
    intersection_score = intersection(geom_a, geom_b)

    area1 = (geom_a.x2 - geom_a.x1) * (geom_a.y2 - geom_a.y1)

    area2 = (geom_b.y2 - geom_b.y1) * (geom_b.x2 - geom_b.x1)

    union_score = area1 + area2 - intersection_score

    overlap_score = intersection_score / union_score

    return overlap_score


def overlap_asym(geom_a, geom_b):
    intersection_score = intersection(geom_a, geom_b)

    area2 = (geom_b.y2 - geom_b.y1) * (geom_b.x2 - geom_b.x1)

    overlap_score = intersection_score / area2

    return overlap_score


def intersection(geom_a, geom_b):
    intersection_score = func.greatest(0, (func.least(geom_a.x2, geom_b.x2) -
                                           func.greatest(geom_a.x1, geom_b.x1))) * \
        func.greatest(0, (func.least(geom_a.y2, geom_b.y2) -
                          func.greatest(geom_a.y1, geom_b.y1)))

    return intersection_score


def union(geom_a, geom_b):
    intersection_score = intersection(geom_a, geom_b)

    area1 = (geom_a.x2 - geom_a.x1) * (geom_a.y2 - geom_a.y1)

    area2 = (geom_b.y2 - geom_b.y1) * (geom_b.x2 - geom_b.x1)

    union_score = area1 + area2 - intersection_score

    return union_score
