#!/usr/bin/env python

import query_utils

# standard imports
import logging
import argparse

# nyc3dcars imports
import nyc3dcars

from celery.task import task

from sqlalchemy import func, or_, desc

import scores


@task
def nms(pid, model, method):
    logging.info((pid))

    try:
        session = nyc3dcars.SESSION()

        scoring_method = scores.METHODS[method]

        set_nms = str(scoring_method.output).split('.')[-1]

        session.query(nyc3dcars.Model) \
            .filter(nyc3dcars.Model.filename == model) \
            .one()

        todo, = session.query(func.count(nyc3dcars.Detection.id)) \
            .join(nyc3dcars.Model) \
            .filter(nyc3dcars.Detection.pid == pid) \
            .filter(or_(*[m == None for m in scoring_method.inputs])) \
            .filter(nyc3dcars.Model.filename == model) \
            .one()

        if todo > 0:
            raise Exception('Some input was not yet computed')

        pos = 0
        while True:
            if pos % 10 == 0:
                logging.info(pos)

            result = session.query(nyc3dcars.Detection) \
                .join(nyc3dcars.Model) \
                .filter(nyc3dcars.Detection.pid == pid) \
                .filter(scoring_method.output == None) \
                .filter(nyc3dcars.Model.filename == model) \

            result = result \
                .order_by(desc(scoring_method.score)) \
                .first()

            if result is None:
                break

            setattr(result, set_nms, True)

            overlap = query_utils.overlap(result, nyc3dcars.Detection)
            covered = overlap > 0.3

            blacklist = session.query(nyc3dcars.Detection) \
                .join(nyc3dcars.Model) \
                .filter(nyc3dcars.Detection.pid == pid) \
                .filter(scoring_method.output == None) \
                .filter(nyc3dcars.Model.filename == model) \
                .filter(covered) \

            for elt in blacklist:
                setattr(elt, set_nms, False)

            pos += 1

        session.commit()

        return pid
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--method', choices=scores.METHODS.keys(), required=True)
    args = parser.parse_args()

    nms(**vars(args))
