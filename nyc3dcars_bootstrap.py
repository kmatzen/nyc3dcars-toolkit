#!/usr/bin/env python

import os
import urllib
import gzip
import tarfile
import subprocess
import argparse

db_manifest = [
    'geodatabase.dump.gz',
    'nyc3dcars.dump.gz',
]

location_manifest = [
    'times-square',
    'apple',
    'macys',
]

image_manifest = ['%s-images.tar.gz'%l for l in location_manifest]

manifest = db_manifest + image_manifest

dbname = 'nyc3dcars'

def create_database (clobberdb):
    if clobberdb:
        print ('Clobbering database')
        subprocess.check_call(['dropdb', dbname])

    subprocess.check_call(['createdb', dbname, '--locale=en_US.utf8', '--encoding=utf8', '-T', 'template0'])
    subprocess.check_call(['psql', dbname, '-c', 'CREATE EXTENSION postgis'])

    srid102718_source = urllib.urlopen('http://spatialreference.org/ref/esri/102718/postgis/')
    srid102718 = srid102718_source.read()
    srid102718 = srid102718.replace('9102718', '102718')

    subprocess.check_call(['psql', dbname, '-c', srid102718])

def download_if_missing (local):
    remote = 'https://s3.amazonaws.com/nyc3dcars/%s'%local

    if not os.path.exists (local):
        print ('Fetching %s'%remote)
        urllib.urlretrieve (remote, local)

def restore_psql_dump (filename):
    print ('Restoring %s'%filename)
    with gzip.open(filename, 'rb') as f:
        proc = subprocess.Popen([
            'pg_restore',
            '--dbname=%s'%dbname,    
        ], stdin=subprocess.PIPE)
        proc.stdin.write(f.read())
        retval = proc.wait()
        if retval != 0:
            raise Exception ('Failed to restore %s'%filename)
            
def extract_archive (filename):
    with tarfile.TarFile (filename, 'r') as f:
        f.extractall()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument ('--clobberdb', action='store_true')
    args = parser.parse_args()

    print ('Downloading resources')
    for filename in manifest:
        download_if_missing (filename)

    print ('Creating database')
    create_database (args.clobberdb)

    print ('Building geodatabase')
    for filename in db_manifest:
        restore_psql_dump (filename)

    print ('Extracting images')
    for filename in image_manifest:
        extract_archive (filename)


