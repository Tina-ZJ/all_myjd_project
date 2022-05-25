# -*-coding=utf-8
import sys

import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os

save_table='sz_algo_terms_weights'
source_table='sz_algo_terms_weights_rank'

def create_table(hiveCtx):
    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            terms    string,
            weights    string,
            count    int
            )
        PARTITIONED BY(dt string, dim_type string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/{table_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(table_name=save_table)
    hiveCtx.sql(create_tbl)



def main_func(rows):
    for row in rows:
        term, cid3_list, weight_list = row
        combine = list()
        for c, w in zip(cid3_list, weight_list):
           combine.append(c+':'+str(w)) 
        yield term, ','.join(combine), len(combine)

def get_weight(hiveCtx):
    sql = """
        select term, collect_list(cid3), collect_list(ctf_brand_igm)
        from {source_table}
        where dt='{dt}' and dim_type='{dim_type}' and rank_igm<=5500 group by term
        """.format(source_table=source_table, dt=dt_str, dim_type=dim_type)


    data = hiveCtx.sql(sql).rdd.cache()
    data.repartition(200).mapPartitions(lambda rows: main_func(rows))\
           .toDF(["term", "weights", "count"]).registerTempTable("temp_table")


    insert_sql = """
        insert overwrite table {table_name} partition(dt='{dt}', dim_type='{dim_type}')
        select  * from temp_table
        """.format(table_name=save_table, dt=dt_str, dim_type=dim_type)
    print("insert_sql:\n" + insert_sql)
    hiveCtx.sql(insert_sql)
    





if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf, appName="sz_algo_recallrelation_query_score_feature")
    sc.setLogLevel("WARN")
    hiveCtx = HiveContext(sc)
    hiveCtx.setConf('spark.shuffle.consolidateFiles', 'true')
    hiveCtx.setConf('spark.shuffle.memoryFraction', '0.4')
    hiveCtx.setConf('spark.sql.shuffle.partitions', '1000')
    if len(sys.argv) == 1:
        dt = datetime.datetime.now() + datetime.timedelta(-1)
    else:
        dt = datetime.datetime.strptime(sys.argv[1], "%Y%m%d").date()


    dt_str = dt.strftime("%Y-%m-%d")
    yest_dt=dt + datetime.timedelta(-1)
    yest_str=yest_dt.strftime("%Y-%m-%d")
    dim_type='crf'
    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_weight(hiveCtx)
