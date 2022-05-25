# -*-coding=utf-8
import sys

import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os

def create_table(hiveCtx):
    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.app_algo_recallrelation_query_sku_cate_score (
            query string,
            sku string,
            sku_name string,
            query_sku_score float,
            query_cate string,
            query_cate_weight string,
            sku_cate string,
            sku_cate_weight string,
            sku_org_cate string,
            search_count int
            )
        PARTITIONED BY(dt string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/app_algo_recallrelation_query_sku_cate_score'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """
    hiveCtx.sql(create_tbl)

def get_score(q_c, sku_c, cid3_dict):
    if q_c in cid3_dict:
        score_list = cid3_dict[q_c].split(',')
        for score in score_list:
            c,s = score.split(':')
            if c == sku_c:
                return float(s)
    return 0
    
def main_func(rows, cid3_dict):
    for row in rows:
        query, sku, sku_name, query_cate, query_cate_weight, sku_cate, sku_cate_weight, sku_org_cate, search_count= row
        score = 0.0
        if query_cate=='' or query_cate==None or sku_cate=='' or sku_cate==None:
            continue
        q_cid = query_cate.split(',')
        q_cid_weight = query_cate_weight.split(',')
        sku_cid = sku_cate.split(',')
        sku_cid_weight = sku_cate_weight.split(',')
        # normlize weight
        q_sum_weight = sum([float(x) for x in q_cid_weight])
        sku_sum_weight = sum([float(x) for x in sku_cid_weight])
        # comput score
        for q_w, q_c in zip(q_cid_weight, q_cid):
            for sku_w, sku_c in zip(sku_cid_weight, sku_cid):
                c2c_score = get_score(q_c, sku_c, cid3_dict)
                score+=(float(q_w)/q_sum_weight)*(float(sku_w)/sku_sum_weight)*float(c2c_score) 
        yield query, sku, sku_name, score, query_cate, query_cate_weight, sku_cate, sku_cate_weight, sku_org_cate, search_count 
 
def getQuery(hiveCtx):
    sql="""
        select query, sku, sku_name, query_cate, query_cate_weigth, sku_cate, sku_cate_weigth, sku_org_cate, search_count
        from app.app_algo_recallrelation_query_sku_cate where dt='2020-05-14' and search_count>2
    """
    print(sql)
    sql_score="""
        select cid3, score_top100
        from app.app_algo_recallrelation_cid_top100_score
        where dt='active'
    """
    print(sql_score)
    ###  cids score data ######
    cid3_data = hiveCtx.sql(sql_score)
    cid3_data_rdd = cid3_data.rdd.map(lambda x : (x[0],x[1]))
    cid3_dict = cid3_data_rdd.collectAsMap()
    #hiveCtx.sql(sql).registerTempTable("tmp_table")
    data = hiveCtx.sql(sql).rdd.cache()
    data.repartition(200).mapPartitions(lambda rows: main_func(rows, cid3_dict))\
        .toDF(["query","sku", "sku_name", "query_sku_score", "query_cate", "query_cate_weight", "sku_cate","sku_cate_weight","sku_org_cate","search_count"]).registerTempTable("tmp_table")

    insert_sql = """
        insert overwrite table app.app_algo_recallrelation_query_sku_cate_score partition(dt='{dt}')
        select  * from tmp_table
        """.format(dt=dt_str)
    print("insert_sql:\n" + insert_sql)
    hiveCtx.sql(insert_sql)



if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf, appName="sp_ind")
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
    yest_dt=dt + datetime.timedelta(-30)
    yest_str=yest_dt.strftime("%Y-%m-%d")

    hiveCtx.sql("use app")
    create_table(hiveCtx)
    getQuery(hiveCtx)

