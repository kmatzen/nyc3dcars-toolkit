#!/bin/bash

mkdir -p nyc3dcars-csv
chmod 777 nyc3dcars-csv

# exporting only datasets Times Square, Macy's, and Apple Store.

psql nyc3dcars -c "drop table photos_export;"
psql nyc3dcars -c "drop table vehicles_export;"
psql nyc3dcars -c "drop table vehicle_types_export;"
psql nyc3dcars -c "drop table databases_export;"
psql nyc3dcars -c "select id, filename, camera_make, camera_model, datetime, focal_length, exposure_time, shutter_speed_value, software, aperture, iso, width, height, focal, k1, k2, r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3, roll, sees_ground, camera_height, test, dataset_id, photographer, flickrid, daytime, ST_AsText(lla) as lla, ST_AsText(geom) as geom into photos_export from photos where dataset_id in (1,2,3);"
psql nyc3dcars -c "select id, pid, x, z, theta, x1, y1, x2, y2, occlusion, view_theta, view_phi, type_id, ST_AsText(lla) as lla, ST_AsText(geom) as geom into vehicles_export from vehicles;"
psql nyc3dcars -c "select * into vehicle_types_export from vehicle_types where id in (202,8,150,123,63,16);"
psql nyc3dcars -c "select * into datasets_export from datasets where id in (1,2,3);"
psql nyc3dcars -c "copy photos_export to '$PWD/nyc3dcars-csv/photos.csv' delimiter ',' csv header;"
psql nyc3dcars -c "copy vehicles_export to '$PWD/nyc3dcars-csv/vehicles.csv' delimiter ',' csv header;"
psql nyc3dcars -c "copy datasets_export to '$PWD/nyc3dcars-csv/datasets.csv' delimiter ',' csv header;"
psql nyc3dcars -c "copy vehicle_types_export to '$PWD/nyc3dcars-csv/vehicle_types.csv' delimiter ',' csv header;"

tar czf nyc3dcars-csv.tar.gz nyc3dcars-csv/

