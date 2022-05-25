# -*-coding=utf-8
import sys
import math
import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os
#import numpy as np

source_table='app_sdl_yinliu_search_click_log'
save_table='sz_algo_query_term_weights_click_labels'


def create_table(hiveCtx):

    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            query                                 string                     COMMENT       'query',
            agg_len                               int                        COMMENT       '聚合的query/sku数量',
            terms                                 array<string>              COMMENT       'query分词',
            avg_weights                          array<string>                COMMENT       'term词权重'
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
        terms_pv = dict()
        query, terms, sku_weight_list, pv_list = row

        # 统计每个term的词频
        for sku_weight, pv in zip(sku_weight_list, pv_list):
            for tw in sku_weight:
                t = tw.f0_term
                w = tw.f1_termweight
                if t not in terms_pv:
                    terms_pv.setdefault(t,0.0)
                terms_pv[t] += float(pv)*w 
        # 计算分值
        weight_list = list()
        for x in terms:
            #weight = terms_pv.get(x, 0.0) /sum(pv_list)
            weight = terms_pv.get(x, 0.0)
            weight_list.append(weight)

        # normalization   
        all_w = sum(weight_list)
        if all_w<0.000001:
            continue
        weight_list = [x/all_w for x in weight_list]
        weight_combine = [t+':'+str(w) for t,w in zip(terms, weight_list)]
        # top 3 
        yield query, len(sku_weight_list), terms, weight_combine
            


def get_sample(hiveCtx):

    
######################### get query segs #############################

    sql = """
                select key_word_qp_norm as query, key_word_qp_c_termlist as terms
                from app.app_m90_search_query_da_qp
                where dt='{dt}' and dim_type='{dim_type}' and key_word_qp_norm is not null
                group by key_word_qp_norm, key_word_qp_c_termlist
        """.format(dt='jd_active', dim_type='all_last_7')

    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_query")


######################### get click ################################

    sql="""
        select sku, key_word, terms, pv
        from
        (
            select query, terms
            from table_query
        )a
        join
        (
            select sku, key_word, count(1) as pv
            from {source_table}
            where dt>'{dt}' and dim_type='app' and trim(key_word)!='' and key_word is not null
            group by sku, key_word
        )b
        on a.query=b.key_word
        """.format(source_table=source_table, dt=week_str)
    
    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_term")


##################### get sku name  #############################

    sql="""
        select key_word, terms, sku_name_qp_c_terminfo as sku_weight, pv
        from
        (
            select sku, key_word, terms, pv
            from table_term
        )a
        join
        (
            select item_sku_id, sku_name_qp_c_terminfo from app.app_m03_search_item_sku_da_qp_simple
            where dt='{dt}' and dim_type='all_last_1'
        )b
        on a.sku=b.item_sku_id
        """.format(dt=dt_str)

    hiveCtx.sql(sql).registerTempTable("table_sku")

    sql="""
        select key_word, terms, collect_list(sku_weight) as sku_weight_list, collect_list(pv) as pv_list
        from table_sku
        group by key_word, terms
        """

    print(sql)
    rdd = hiveCtx.sql(sql).rdd.cache()
    rdd.repartition(500).mapPartitions(lambda rows: score_func(rows))\
        .toDF(['query','agg_len','terms','weights']).registerTempTable("score_table")
   

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

    dt_str = dt.strftime("%Y-%m-%d")
    week_dt=dt + datetime.timedelta(-2)
    week_str=week_dt.strftime("%Y-%m-%d")
    dim_type='query_v3'
    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_sample(hiveCtx)


