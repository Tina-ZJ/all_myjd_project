# -*-coding=utf-8
import sys
import math
import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os
#import numpy as np

source_table='app_sdl_yinliu_search_click_log'
save_table='sz_algo_sku_term_weights_click_labels'


def create_table(hiveCtx):

    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            sku                                   string                     COMMENT   'sku',
            agg_len                               int                        COMMENT   '聚合的query数量',
            agg_querys                            array<array<string>>       COMMENT   '聚合的query集合',
            terms                                 array<string>              COMMENT   'sku分词',
            weight                                array<string>              COMMENT   'term词权重平均值'
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
        sku, sku_seg, query_seg_list, pv_list = row

        # 统计每个term的词频
        for terms, pv in zip(query_seg_list, pv_list):
            for tw in terms:
                x = tw.f0_term
                w = tw.f1_termweight
                if x not in terms_pv:
                    terms_pv.setdefault(x,0.0)
                #terms_pv[x] += float(pv)*w
                terms_pv[x] += float(pv)

        # 计算分值
        pv_sum = sum(pv_list)
        weight_list = list()
        sku_weight_list = list()
        seg_list = list()
        for tw in sku_seg:
            #weight = terms_pv.get(term, 0.0) / pv_sum
            term = tw.f0_term
            seg_list.append(term)
            sw = tw.f1_termweight
            weight = terms_pv.get(term, 0.0)
            weight_list.append(weight)
            sku_weight_list.append(sw)
        # normalization
        score_sum = sum(weight_list)
        sku_score = sum(sku_weight_list)
        if score_sum < 0.000001 or sku_score < 0.00001:
            continue

        weight_list = [x/score_sum for x in weight_list]
        all_weight_list = [ 0.4*sw+0.6*tw for sw, tw in zip(sku_weight_list, weight_list)]
        all_score_sum = sum(all_weight_list)
        weight_list = [x/all_score_sum for x in all_weight_list]
        weight_list = [t.f0_term+':'+str(x) for t,x in zip(sku_seg, weight_list)]
        
        yield sku, len(query_seg_list), query_seg_list[:10], seg_list, weight_list
            


def get_sample(hiveCtx):

    
######################### get query cid3 label #############################

    sql = """
                select key_word_qp_norm as query, key_word_qp_c_terminfo as terms
                from app.app_m90_search_query_da_qp
                where dt='{dt}' and dim_type='{dim_type}' and key_word_qp_norm is not null
                group by key_word_qp_norm, key_word_qp_c_terminfo
        """.format(dt='jd_active', dim_type='all_last_7')

    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_query")



    sql="""
        select sku, key_word, sku_seg, pv
        from
        (
            select sku, key_word, count(1) as pv
            from {source_table}
            where dt>'{dt_week}' and dim_type='app' and trim(key_word)!='' and key_word is not null
            group by sku, key_word
        )a
        join
        (
            select item_sku_id, sku_name_qp_c_terminfo as sku_seg 
            from app.app_m03_search_item_sku_da_qp_simple
            where dt='jd_active' and dim_type='all_last_1' 
        )b
        on a.sku=b.item_sku_id
        """.format(source_table=source_table, dt_week=week_str, dt=dt_str)
    
    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_sku")

    sql="""
        select sku, sku_seg, terms, pv
        from
        (
            select query, terms 
            from table_query
        )a
        join
        (
            select sku, key_word, sku_seg, pv
            from table_sku
        )b
        on a.query=b.key_word
        """

    print(sql)
    hiveCtx.sql(sql).registerTempTable("sku_query_table")

    sql="""
        select sku, sku_seg, collect_list(terms) as query_seg_list, collect_list(pv) as pv_list
        from sku_query_table
        group by sku, sku_seg
        """
    print(sql)
    data = hiveCtx.sql(sql).rdd.cache()
    data.repartition(500).mapPartitions(lambda rows: score_func(rows))\
        .toDF(['sku', 'len_querys','querys_list','sku_seg', 'weights' ]).registerTempTable("weights_table")


    insert_sql = """
                insert overwrite table {save_table} partition(dt='{dt}', dim_type='{dim_type}')
                select * from weights_table 
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
    week_dt=dt + datetime.timedelta(-7)
    week_str=week_dt.strftime("%Y-%m-%d")
    dim_type='v3'
    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_sample(hiveCtx)


