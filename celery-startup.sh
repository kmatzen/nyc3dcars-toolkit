#!/bin/bash

hostname=`hostname`

mkdir -p celery/logs
mkdir -p celery/pids

for i in {0..27}
do
    host=`printf %s%02d $hostname $i`
    echo $host
    # pydro is multithreaded, but when so many instances are running in parallel, 
    # we found that it worked best to make each one use only one thread.
    export OMP_NUM_THREADS=1 
    nohup celery worker -l info -n $host -f celery/logs/$host.log -E --pidfile=celery/pids/$host.pid 2> /dev/null &
done
