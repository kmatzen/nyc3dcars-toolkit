#!/usr/bin/env python

import argparse
import math

import nyc3dcars

import pydro.io

def register_model (filename, a, b, thresh, viewpoint):
    session = nyc3dcars.Session()

    if viewpoint:
        model = pydro.io.LoadModel(filename)

        for i in xrange(16):
            model.start.rules[i].metadata = {'angle':(math.pi - i*math.pi/8)%(2*math.pi)}

        pydro.io.SaveModel(filename, model)

    model = nyc3dcars.Model(
        filename=filename,
        a=a,
        b=b,
        thresh=thresh,
        release='pydro',
    )
    session.add(model)
    session.commit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True)
    parser.add_argument('--a', required=True, type=float)
    parser.add_argument('--b', required=True, type=float)
    parser.add_argument('--thresh', required=True, type=float)
    parser.add_argument('--viewpoint', action='store_true')
    args = parser.parse_args()

    register_model(**vars(args))
