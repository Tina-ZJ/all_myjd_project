#!/bin/bash

# set save data days
days=10

now=$(date +%s)
hadoop fs -ls -r hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/qp_personlization/ | while read f; do
    # get dir time
    dir_date=`echo $f | awk '{print $6}'`
    # get time gap (day)
    day_gap=$(( ( $now - $(date -d "$dir_date" +%s) ) / (24 * 60 * 60 ) ))
    # delelte old tfrecord data 
    if [ $day_gap -gt $days ]; then
        dir=`echo $f | awk '{print $8}'`
        hadoop fs -rm -r $dir;
    fi
done
