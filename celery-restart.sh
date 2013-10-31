#!/bin/bash

find celery/pids/ -type f -exec cat {} \;|xargs kill -HUP
