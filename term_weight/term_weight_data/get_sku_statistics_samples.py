# -*-coding=utf-8
import sys
import math
import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os
#import numpy as np

source_table='sz_algo_cid3_words_statistics_features'
source_table2='sz_algo_sku_term_weights_click_labels'
save_table='sz_algo_sku_term_weights_label'


def create_table(hiveCtx):

    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            sku                                 string                     COMMENT   'sku id',
            terms                                 array<string>              COMMENT   'query分词',
            weight                                 array<string>              COMMENT   'weight',
            cid3                                  string                     COMMENT   'cid3',
            cid3_name                             string                     COMMENT   'cid3 名称',
            term_features                         map<string, array<float>>  COMMENT   '每个term对应的ctf, brand, ctf_confidence, rf, icf, igm特征'
            )
        PARTITIONED BY(dt string, dim_type string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/{table_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(table_name=save_table)

    print(create_tbl)
    hiveCtx.sql(create_tbl)
  


def score_func(rows):
    for row in rows:
        
        sku, terms, weight, cid3, cid3_name, term_list, brand_list, ctf_list, confidence_list, rf_list, icf_list, igm_list = row
        score_list = list()
        if len(term_list)!=len(brand_list)!=len(ctf_list)!=len(confidence_list)!=len(icf_list)!=len(igm_list):
            continue
        terms_f = dict()
        for term, brand, ctf, confidence, rf, icf, igm in zip(term_list, brand_list, ctf_list, confidence_list, rf_list, icf_list, igm_list):
            brand = float(brand)
            
            # save term features
            terms_f.setdefault(term, [brand, ctf, confidence, rf, icf, igm*100])
        
        yield sku, terms, weight, cid3, cid3_name, terms_f


def get_sample(hiveCtx):

    
######################### get sku cid3 label #############################

    sql = """
                select sku, terms, weight, cid3, cid3_name
                from
                (
                    select sku, terms, weight  
                    from {source_table2}
                    where dt='{dt}' and dim_type='{dim_type}' and agg_len >10
                )a
                join
                (
                    select item_sku_id, item_last_cate_cd as cid3, item_last_cate_name as cid3_name
                    from app.app_m03_search_item_sku_da_qp_simple where dt='jd_active' and version='jd_active' and dim_type='all_last_1'
                )b
                on a.sku=b.item_sku_id
        """.format(source_table2=source_table2, dt=dt_str, dim_type=dim_type)

    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_sku")



    sql="""
        select sku, terms, weight, a.cid3 as cid3, cid3_name, a.term as term, brand_flag, ctf, ctf_confidence, rf, icf, igm
        from
        (
            select sku, terms, weight, cid3, cid3_name, term 
            from table_sku
            LATERAL VIEW explode(terms) t as term
        )a
        join
        (
            select cid3, term, brand_flag, ctf, ctf_confidence, rf, icf, igm
            from {source_table}
            where dt='{dt}' and dim_type='{dim_type}'
        )b
        on a.cid3=b.cid3 and a.term=b.term
        """.format(source_table=source_table, dt='2021-12-05', dim_type='crf')
    
    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_term")

    sql="""
        select sku, terms, weight, cid3, cid3_name, collect_list(term) as term_list, collect_list(brand_flag) as brand_list, collect_list(ctf) as ctf_list, 
        collect_list(ctf_confidence) as confidence_list, collect_list(rf) as rf_list, collect_list(icf) as icf_list, collect_list(igm) as igm_list
        from table_term
        group by sku, terms, weight, cid3, cid3_name
        """

    print(sql)
    rdd = hiveCtx.sql(sql).rdd.cache()
    rdd.repartition(1000).mapPartitions(lambda rows: score_func(rows))\
        .toDF(['sku', 'terms', 'weight', 'cid3', 'cid3_name', 'term_features']).registerTempTable("score_table")
    
    insert_sql = """
                insert overwrite table {save_table} partition(dt='{dt}', dim_type='{dim_type}')
                select * from score_table
                """.format(save_table=save_table, dt=dt_str, dim_type=dim_type)

    print("insert_sql:\n" + insert_sql)
    hiveCtx.sql(insert_sql)
    









if __name__=='__main__':
    conf = SparkConf()
    sc = SparkContext(conf=conf, appName="term_weight_ndcg_score")
    sc.setLogLevel("WARN")
    hiveCtx = HiveContext(sc)
    hiveCtx.setConf('spark.shuffle.consolidateFiles', 'true')
    hiveCtx.setConf('spark.shuffle.memoryFraction', '0.4')
    hiveCtx.setConf('spark.sql.shuffle.partitions', '1000')
    if len(sys.argv) == 1:
        dt = datetime.datetime.now() + datetime.timedelta(-1)
    else:
        dt = datetime.datetime.strptime(sys.argv[1], "%Y%m%d").date()

    dim_type='v4'
    dt_str = dt.strftime("%Y-%m-%d")
    yest_dt=dt + datetime.timedelta(-1)
    yest_str=yest_dt.strftime("%Y-%m-%d")

    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_sample(hiveCtx)


