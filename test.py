#!/usr/bin/env python

import celery

import nyc3dcars

import scores

import logging
import argparse

from detect import detect
from geo_rescore import geo_rescore
from nms import nms


def flatten(task):
    if task is None:
        return None
    else:
        parent = flatten(task.parent)
        if parent is None:
            return [task.task_id]
        else:
            return [task.task_id] + parent


def test(model, remote, methods, dataset_id):
    session = nyc3dcars.SESSION()
    try:
        test_set = session.query(nyc3dcars.Photo) \
            .filter(nyc3dcars.Photo.test == True) \
            .filter(nyc3dcars.Photo.dataset_id == dataset_id) \

        session.query(nyc3dcars.Model) \
            .filter_by(filename=model) \
            .one()

        for photo in test_set:
            logging.info('PID: %d' % photo.id)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--methods', nargs='+', default=scores.METHODS.keys())
    parser.add_argument('--dataset-id', required=True, type=int)
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()

    test(**vars(args))
