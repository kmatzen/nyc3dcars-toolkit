from sqlalchemy import *

def overlap(geomA, geomB):
    intersection_score = intersection(geomA, geomB)

    area1 = (geomA.x2 - geomA.x1) * (geomA.y2 - geomA.y1)

    area2 = (geomB.y2 - geomB.y1) * (geomB.x2 - geomB.x1)

    union_score = area1 + area2 - intersection_score

    overlap_score = intersection_score/union_score

    return overlap_score

def overlap_asym(geomA, geomB):
    intersection_score = intersection(geomA, geomB)

    area2 = (geomB.y2 - geomB.y1) * (geomB.x2 - geomB.x1)

    overlap_score = intersection_score/area2

    return overlap_score

def intersection(geomA, geomB):
    intersection_score = func.greatest(0, (func.least(geomA.x2, geomB.x2) - 
                                     func.greatest(geomA.x1, geomB.x1))) * \
                   func.greatest(0, (func.least(geomA.y2, geomB.y2) -
                                     func.greatest(geomA.y1, geomB.y1)))

    return intersection_score

def union(geomA, geomB):
    intersection_score = intersection(geomA, geomB)

    area1 = (geomA.x2 - geomA.x1) * (geomA.y2 - geomA.y1)

    area2 = (geomB.y2 - geomB.y1) * (geomB.x2 - geomB.x1)

    union_score = area1 + area2 - intersection_score

    return union_score
