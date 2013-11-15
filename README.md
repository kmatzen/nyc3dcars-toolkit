nyc3dcars-toolkit
=================

PostgreSQL and PostGIS are both required for these scripts.  If you're using a recent version of Ubuntu, then installing PostgreSQL is as simple as
```
sudo apt-get install postgresql-9.1
```

However, you should probably use this special PPA to install PostGIS, which can be done by running
```
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:ubuntugis/ppa
```
and then do
```
sudo apt-get install postgis
```

If you're going to use the nyc3dcars\_bootstrap.py script, then make a PostgreSQL user with the same username as your current user.  If you just recently installed PostgreSQL from Ubuntu's repository, then there's a default user named postgres.  Execute the following, granting super user privileges when prompted.
```
sudo -u postgres createuser <username>
```

I have a couple projects that are necessary to run this code.
https://github.com/kmatzen/pydro.git
https://github.com/kmatzen/pygeo.git
Pydro is an implementation of the deformable part model in C and Python.  The detection routines are functionally equivalent to Pedro Felzenszwalb's voc-release5 and the training procedure is still in development.  Right now it offers a slight performance improvement on some examples and is somewhat easier to integrate into a larger system.
Pygeo is a collection of transformations for working with geographic data.  In particular it can convert between Earth-Center Earth-Fixed (ECEF) cartesian coordinates and WGS84 ellipsoidal latitude, longitude, and altitude as well as get a local East-North-Up rotation matrix for a particular latitude and longitude.

I usually run the test.py script with the --remote flag which http://www.celeryproject.org/ to distribute the computation on a cluster.  If you don't supply this flag, it will run everything with one thread.  Even if you don't want to run the testing protocol on a cluster, I still recommend remote mode to enable parallel computation.  It only takes a couple of steps to get working.  One is to use a Celery backend.  I use RabbitMQ.  Instructions on installing RabbitMQ for Ubuntu can be found here http://www.rabbitmq.com/install-debian.html.  The second step is to run a celery worker on each node.  I've supplied an example celeryconfig.py.  To run a worker, execute
```
celery worker -l info -c <concurrency>
```
On a 16 core node with hyperthreading, I use a concurrency level of 28 (I leave some space for the postgres process).  Note, the following config line is very necessary for correct operation with my code.  Otherwise, celery will fork, but not exec, the connections to the database will be shared, and messages will be clobbered.
```
CELERYD_FORCE_EXECV = True
```

The testing procedure is built like a pipeline.  There are three stages represented by the following scripts:
* detect.py - Runs the DPM, gathers detections, computes 3D vehicle poses, and logs them to the database.
* geo\_rescore.py - Uses scoring methods defined in scores.py to compute additional detections scores related to geographic data.
* nms.py - Runs non-maxima suppression to get the final set of detections.

Each of these scripts can be run individually on specific photos or the test.py script can be used to execute the entire pipeline for all photos in a particular dataset.

gen\_results.py is used to compute precision/orientation similarity and recall curves and gen\_dist\_error.py is used to compute 3D pose localization error.
