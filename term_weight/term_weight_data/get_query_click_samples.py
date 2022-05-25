# -*-coding=utf-8
import sys
import math
import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os
#import numpy as np

source_table='app_sdl_yinliu_search_click_log'
save_table='sz_algo_term_weights_click_labels'


def create_table(hiveCtx):

    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            query                                 string                     COMMENT   'query',
            agg_len                               int                        COMMENT   '聚合的query数量',
            agg_querys                            array<string>              COMMENT   '聚合的query集合',
            terms                                 array<string>              COMMENT   'query分词',
            avg_weight                          array<string>                COMMENT   'term词权重平均值'
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
        sku, pv_list, query_list, terms_list = row

        # 统计每个term的词频
        for terms, pv in zip(terms_list, pv_list):
            for x in terms:
                if x not in terms_pv:
                    terms_pv.setdefault(x,0.0)
                terms_pv[x] += float(pv) 
        # 计算分值
        for query, terms, pv in zip(query_list, terms_list, pv_list):
            weight_list = list()
            for x in terms:
               weight = terms_pv.get(x, 0.0) /sum(pv_list)
               weight_list.append(weight)
            yield sku, query, terms, pv, weight_list, query_list
            

def score_norm_func(rows):
    for row in rows:
        query, terms_list, pv_list, weights_list = row
        weights_avg = [0.0] * len(weights_list[0])
         
        maxs = 0
        #对应位置的term weight相加
        for weights, pv in zip(weights_list, pv_list):
            # 保留最大pv的weight 
            #if pv >maxs:
            #    weights_max = weights
            #    maxs = pv
            for i, w in enumerate(weights):
                weights_avg[i]+=w
        # 求平均
        weights_avg = [ x/len(weights_list) for x in weights_avg ]
        # combine
        t_w = list()
        # normalization
        sums = sum(weights_avg)
        weights_avg = [x/sums for x in weights_avg]

        for t, w in zip(terms_list[0], weights_avg):
            t_w.append(t+':'+str(w))

        yield query, terms_list[0], t_w

def get_sample(hiveCtx):

    
######################### get query cid3 label #############################

    sql = """
                select key_word_qp_norm as query, key_word_qp_c_termlist as terms
                from app.app_m90_search_query_da_qp
                where dt='{dt}' and dim_type='{dim_type}' and key_word_qp_norm is not null
                group by key_word_qp_norm, key_word_qp_c_termlist
        """.format(dt='jd_active', dim_type='all_last_7')

    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_query")



    sql="""
        select sku, key_word, terms, pv
        from
        (
            select query, terms
            from table_query
            group by query, terms
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

    sql="""
        select sku, collect_list(pv) as pv_list, collect_list(key_word) as key_word_list, collect_list(terms) as terms_list 
        from
        (
            select sku, key_word, terms, pv, row_number() over(partition by sku order by pv desc) as rank
            from table_term
        )
        where rank<=100
        group by sku
        """

    print(sql)
    rdd = hiveCtx.sql(sql).rdd.cache()
    rdd.repartition(500).mapPartitions(lambda rows: score_func(rows))\
        .toDF(['sku','query','terms','pv', 'weights', 'query_list']).registerTempTable("score_table")
   

    sql="""
        select query, collect_list(terms) as terms_list, collect_list(pv) as pv_list, collect_list(weights) as weights_list
        from score_table
        group by query
        """
    print(sql)
    data = hiveCtx.sql(sql).rdd.cache()
    data.repartition(500).mapPartitions(lambda rows: score_norm_func(rows))\
        .toDF(['query','terms', 'weights' ]).registerTempTable("weights_table")

    sql="""
            select b.query as query, agg_len, query_list, terms, weights
            from
            (
                select query, size(query_list) as agg_len, query_list 
                from
                (
                select query, pv, query_list, row_number() over(partition by query order by pv desc) as rank
                from score_table
                )
                where rank=1
            )a
            join
            (
               select query, terms, weights
               from weights_table
            )b
            on a.query=b.query
            where agg_len>3
        """
    print(sql)
    hiveCtx.sql(sql).registerTempTable("result_table")

    insert_sql = """
                insert overwrite table {save_table} partition(dt='{dt}', dim_type='{dim_type}')
                select * from result_table 
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
    dim_type='v1'
    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_sample(hiveCtx)


