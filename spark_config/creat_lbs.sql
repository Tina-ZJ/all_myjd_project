use app;
set mapred.job.priority=VERY_HIGH;
set mapreduce.job.queuename=bdp_jmart_recsys.recsys_suggest;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.dynamic.partition=True;
CREATE EXTERNAL TABLE IF NOT EXISTS `app.app_test_lbs`(
    `query` string COMMENT 'query',
    `segs` string COMMENT '分词',
    `lbs` string COMMENT 'lbs意图')
PARTITIONED BY (
    `dt` string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
STORED AS textfile
LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/'
TBLPROPERTIES (
    'orc.compress'='SNAPPY',
    'SENSITIVE_TABLE'='FALSE');
