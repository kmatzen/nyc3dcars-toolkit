#!/usr/bin/env python

"""Runs the testing protocol for a given dataset, model, and NMS methods.
   Can either run as a single process or distributed using celery."""

import celery

import nyc3dcars

import scores

import logging
import argparse

from detect import detect
from geo_rescore import geo_rescore
from nms import nms


def test(model, remote, methods, dataset_id):
    """Executes the testing protocol."""

    session = nyc3dcars.SESSION()
    try:
        test_set = session.query(nyc3dcars.Photo) \
            .filter_by(test=True, dataset_id=dataset_id)

        session.query(nyc3dcars.Model) \
            .filter_by(filename=model) \
            .one()

        for photo in test_set:
            logging.info(photo.id)

            celery_list = [detect.s(photo.id, model)]

            for method in methods:
                celery_list += [geo_rescore.s(model, method)]

            for method in methods:
                celery_list += [nms.s(model, method)]

            celery_task = celery.chain(celery_list)

            if remote:
                celery_task.apply_async()
            else:
                celery_task.apply()

    except:
        session.rollback()
        raise

    finally:
        session.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--model', required=True)
    PARSER.add_argument('--methods', nargs='+', default=scores.METHODS.keys())
    PARSER.add_argument('--dataset-id', required=True, type=int)
    PARSER.add_argument('--remote', action='store_true')
    ARGS = PARSER.parse_args()

    test(
        model=ARGS.model,
        remote=ARGS.remote,
        methods=ARGS.methods,
        dataset_id=ARGS.dataset_id,
    )
