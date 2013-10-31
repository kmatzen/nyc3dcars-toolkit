#!/usr/bin/env python

"""Computes AP and AOS for filtered detection results."""

from query_utils import precision_recall, orientation_similarity

import scores

import logging
import itertools
import scipy.integrate
import numpy
import nyc3dcars
import argparse
from sqlalchemy import func, and_



def gen_results(model, methods, aos, dataset_id):
    """Computes PR curve and optionally OS-R curve."""

    session = nyc3dcars.SESSION()
    try:

        difficulties = {
            'full': []
        }

        daynights = {
            'both': [],
        }

        # pylint: disable-msg=E1101
        model_id, = session.query(nyc3dcars.Model.id) \
            .filter(nyc3dcars.Model.filename == model) \
            .one()

        todo, = session.query(func.count(nyc3dcars.Photo.id)) \
            .outerjoin((
                nyc3dcars.Detection,
                and_(
                    nyc3dcars.Detection.pid == nyc3dcars.Photo.id,
                    nyc3dcars.Detection.pmid == model_id
                )
            )) \
            .filter(nyc3dcars.Photo.test == True) \
            .filter(nyc3dcars.Detection.id == None) \
            .filter(nyc3dcars.Photo.dataset_id == dataset_id) \
            .one()
        # pylint: enable-msg=E1101

        if todo > 0:
            msg = '%s is not ready. %d photos remaining' % (model, todo)
            logging.info(msg)
            return

        not_ready = False

        for name in methods:
            nms_method = scores.METHODS[name]
            # pylint: disable-msg=E1101
            todo, = session.query(func.count(nyc3dcars.Detection.id)) \
                .join(nyc3dcars.Model) \
                .join(nyc3dcars.Photo) \
                .filter(nyc3dcars.Photo.test == True) \
                .filter(nyc3dcars.Model.filename == model) \
                .filter(nyc3dcars.Photo.dataset_id == dataset_id) \
                .filter(nms_method.output == None) \
                .one()
            # pylint: enable-msg=E1101
            if todo > 0:
                msg = '%s is not ready.  %d %s NMS remaining' % (
                    model, todo, name)
                logging.info(msg)
                not_ready = True

        if not_ready:
            return

        # pylint: disable-msg=E1101
        dataset_id = [nyc3dcars.Photo.dataset_id == dataset_id]
        # pylint: enable-msg=E1101

        for daynight, difficulty in itertools.product(daynights, difficulties):
            for method in methods:
                nms_method = scores.METHODS[method]
                selected = [nms_method.output == True]
                msg = '%s daynight: %s, difficulty: %s, method: %s' % (
                    model, daynight, difficulty, method)
                logging.info(msg)
                points = precision_recall(
                    session,
                    nms_method.score,
                    dataset_id + daynights[daynight] + selected,
                    dataset_id +
                    daynights[daynight] + difficulties[difficulty],
                    model
                )
                name = '%s, %s' % (model, method)
                if aos:
                    points_aos = orientation_similarity(session,
                                                        nms_method.score,
                                                        dataset_id +
                                                        daynights[
                                                            daynight] + selected,
                                                        dataset_id +
                                                        daynights[daynight] + difficulties[
                                                            difficulty],
                                                        model
                                                        )
                else:
                    points_aos = None

                print(points.shape)
                print(scipy.integrate.trapz(points[:, 0], points[:, 1]))
                filename = '%s-%s-%s-%s-pr.txt' % (
                    model, daynight, difficulty, method)
                print(filename)
                numpy.savetxt(filename, points)
                if points_aos is not None:
                    print(scipy.integrate.trapz(
                        points_aos[:, 0], points_aos[:, 1]))
                    filename = '%s-%s-%s-%s-aos.txt' % (
                        model, daynight, difficulty, method)
                    print(filename)
                    numpy.savetxt(filename, points_aos)

        logging.info('done')

    except:
        raise
    finally:
        session.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--model', required=True)
    PARSER.add_argument('--aos', action='store_true')
    PARSER.add_argument('--dataset-id', required=True, type=int)
    PARSER.add_argument('--methods', nargs='+', default=scores.METHODS.keys())
    ARGS = PARSER.parse_args()

    gen_results(
        model=ARGS.model,
        methods=ARGS.methods,
        aos=ARGS.aos,
        dataset_id=ARGS.dataset_id,
    )
