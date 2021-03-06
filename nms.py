#!/usr/bin/env python

"""Task that preforms NMS on detections."""

import query_utils

# standard imports
import logging
import argparse

# nyc3dcars imports
from nyc3dcars import SESSION, Model, Detection

from celery.task import task

from sqlalchemy import func, or_, desc

import scores


@task
def nms(pid, model, method):
    """Preforms NMS on detections."""

    session = SESSION()

    logging.info((pid, model, method))

    try:

        scoring_method = scores.METHODS[method]

        set_nms = str(scoring_method.output).split('.')[-1]

        # pylint: disable-msg=E1101
        mid, = session.query(Model.id) \
            .filter_by(filename=model) \
            .one()

        todo, = session.query(func.count(Detection.id)) \
            .filter(Detection.pid == pid) \
            .filter(or_(*[m == None for m in scoring_method.inputs])) \
            .filter(Detection.mid == mid) \
            .one()
        # pylint: enable-msg=E1101

        if todo > 0:
            raise Exception('Some input was not yet computed')

        while True:
            # pylint: disable-msg=E1101
            result = session.query(Detection) \
                .filter(Detection.pid == pid) \
                .filter(scoring_method.output == None) \
                .filter(Detection.mid == mid)
            # pylint: enable-msg=E1101

            result = result \
                .order_by(desc(scoring_method.score)) \
                .first()

            if result is None:
                break

            setattr(result, set_nms, True)

            overlap = query_utils.overlap(result, Detection)
            covered = overlap > 0.3

            # pylint: disable-msg=E1101
            blacklist = session.query(Detection) \
                .filter(Detection.pid == pid) \
                .filter(scoring_method.output == None) \
                .filter(Detection.mid == mid) \
                .filter(covered)
            # pylint: enable-msg=E1101

            for elt in blacklist:
                setattr(elt, set_nms, False)

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
    PARSER.add_argument(
        '--method',
        choices=scores.METHODS.keys(),
        required=True
    )
    ARGS = PARSER.parse_args()

    nms(
        pid=ARGS.pid,
        model=ARGS.model,
        method=ARGS.method,
    )
