#!/bin/sh

sql_file=$1
data_file=$2
table_name=$3

#create table
hive -f ${sql_file}

hive -e """
use app;
load data local inpath '${data_file}'
overwrite into table ${table_name}
partition(dt='active');
"""
