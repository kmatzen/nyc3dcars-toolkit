#!/usr/bin/env python

import scores

# standard imports
import logging
import argparse
import math

# nyc3dcars imports
import nyc3dcars

# math imports
import numpy

from celery.task import task

from sqlalchemy import func, or_
#from sqlalchemy.orm import *


@task
def geo_rescore(pid, model, method):
    logging.info('geo_rescore %d %s %s' % (pid, model, method))
    session = nyc3dcars.SESSION()
    try:
        numpy.seterr(all='raise')

        session.query(nyc3dcars.Model) \
            .filter(nyc3dcars.Model.filename == model) \
            .one()

        nms_method = scores.METHODS[method]

        detections = session.query(nyc3dcars.Detection) \
            .join(nyc3dcars.Model) \
            .filter(nyc3dcars.Detection.pid == pid) \
            .filter(nyc3dcars.Model.filename == model) \
            .filter(or_(*[m == None for m in nms_method.inputs]))

        nms_method = scores.METHODS[method]

        for method_input in nms_method.inputs:
            score_name = str(method_input).split('.')[-1]
            score = scores.SCORES[score_name]

            if score.compute is None:
                continue

            for detection in detections:
                value = score.compute(session, detection)

                existing = getattr(detection, score_name)

                if existing is not None:
                    if not math.fabs(existing - value) < 1e-8:
                        logging.info(
                            '%s %f %f' % (score_name, existing, value))
                        assert False

                setattr(detection, score_name, value)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return pid

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--method', required=True, choices=scores.METHODS.keys())
    args = parser.parse_args()

    geo_rescore(**vars(args))
