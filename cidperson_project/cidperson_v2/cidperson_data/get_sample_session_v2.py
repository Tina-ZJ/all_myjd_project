# coding=utf-8
import sys

import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os

table_name='app_personalization_cid_samples'
def create_table(hiveCtx):
    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            pvid string,
            keyword string,
            gender_realt array<int>,
            cids_realt array<int>,
            brands_realt array<int>,
            click_cid array<int>,
            age_realt array<int>,
            search_keyword_seg_id array<int>,
            cidx string,
            cidx_name string
            )
        PARTITIONED BY(dt string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/{table_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(table_name=table_name)

    print("create table sql:\n"+create_tbl)
    hiveCtx.sql(create_tbl)

def main_func(rows):
    for row in rows:
        int_feature = row.int_feature
        gender_realt = int_feature['gender_realt']
        age_realt = int_feature['age_realt']
        search_keyword_seg_id = int_feature['search_keyword_seg_id']
        
        yield row.pvid, row.keyword, gender_realt, age_realt, search_keyword_seg_id, row.cidx, row.cidx_name
 
def getSample(hiveCtx):
    sql="""
            select a.pvid as pvid, keyword, gender_realt, cids_realt, brands_realt, click_cid, age_realt, search_keyword_seg_id, sku
            from 
            (
              select  split(ext['pvid'],'_')[0] as pvid, key_word, sku
              from app.app_sdl_yinliu_search_click_log where dt>='{yest_str}' and element_at(ext,'pvid') is not null group by split(ext['pvid'],'_')[0],key_word, sku
            )b 
            join
            (
              select pvid,  keyword, int_feature['gender_realt'] as gender_realt, int_feature['cids_realt'] as cids_realt, int_feature['brands_realt'] as brands_realt, int_feature['click_cid'] as click_cid, int_feature['age_realt'] as age_realt, int_feature['search_keyword_seg_id'] as search_keyword_seg_id 
              from app.user_query_feature_dump_szalgo where dt>='{yest_str}' and keyword!='' group by pvid, keyword, int_feature['gender_realt'], int_feature['cids_realt'], int_feature['brands_realt'], int_feature['click_cid'], int_feature['age_realt'], int_feature['search_keyword_seg_id']
            )a
            on a.pvid=b.pvid and a.keyword=b.key_word
    """.format(yest_str=yest_str)
    print("data_sql:\n" + sql)

    hiveCtx.sql(sql).registerTempTable("tmp_table")

    sql_cid="""
            select pvid, keyword, gender_realt, cids_realt, brands_realt, click_cid, age_realt, search_keyword_seg_id, item_last_cate_cd as cidx, item_last_cate_name as cidx_name 
            from
            (
               select pvid, keyword, gender_realt, cids_realt, brands_realt, click_cid, age_realt, search_keyword_seg_id, sku
               from tmp_table
            )a
            join
            (
              select item_sku_id, item_last_cate_cd, item_last_cate_name
              from gdm.gdm_m03_search_item_sku_da where dt='{dt}' and sku_valid_flag=1 group by item_sku_id, item_last_cate_cd, item_last_cate_name
            )b
            on a.sku=b.item_sku_id
            group by pvid, keyword, gender_realt, cids_realt, brands_realt, click_cid, age_realt, search_keyword_seg_id, item_last_cate_cd, item_last_cate_name
        """.format(dt=yest_str)
 
    print("sql_sql:\n" + sql_cid)
    hiveCtx.sql(sql_cid).registerTempTable("tmp2_table")
    #data = hiveCtx.sql(sql_cid).rdd.cache()
    #data.repartition(200).mapPartitions(lambda rows: main_func(rows))\
    #    .toDF(['pvid','keyword','gender_realt','age_realt','search_keyword_seg_id','cidx','cidx_name']).registerTempTable("tmp2_table")

    insert_sql = """
        insert overwrite table app.{table_name} partition(dt='{dt}')
        select * from tmp2_table
        """.format(table_name=table_name, dt=dt_str)
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
    yest_dt=dt + datetime.timedelta(-15)
    yest_str=yest_dt.strftime("%Y-%m-%d")

    hiveCtx.sql("use app")
    create_table(hiveCtx)
    getSample(hiveCtx)

