#!/usr/bin/env python

"""Contains task to perform geographic rescoring of detections."""

import scores

# standard imports
import logging
import argparse
import math

# nyc3dcars imports
from nyc3dcars import SESSION, Model, Detection

# math imports
import numpy

from celery.task import task

from sqlalchemy import or_


@task
def geo_rescore(pid, model, method):
    """Apply geographic rescoring."""

    logging.info(str((pid, model, method)))
    session = SESSION()
    try:
        numpy.seterr(all='raise')

        session.query(Model) \
            .filter_by(filename=model) \
            .one()

        nms_method = scores.METHODS[method]

        # pylint: disable-msg=E1101
        detections = session.query(Detection) \
            .join(Model) \
            .filter(Detection.pid == pid) \
            .filter(Model.filename == model) \
            .filter(or_(*[m == None for m in nms_method.inputs]))
        # pylint: enable-msg=E1101

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

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--pid', type=int, required=True)
    PARSER.add_argument('--model', required=True)
    PARSER.add_argument(
        '--method',
        required=True,
        choices=scores.METHODS.keys()
    )
    ARGS = PARSER.parse_args()

    geo_rescore(
        pid=ARGS.pid,
        model=ARGS.model,
        method=ARGS.method,
    )
