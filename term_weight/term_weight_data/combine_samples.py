# -*-coding=utf-8
import sys
import math
import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os
#import numpy as np

click_table='sz_algo_term_weights_click_labels'
statistics_table='sz_algo_query_term_weights_label'
save_table='sz_algo_query_term_weights_samples'


def create_table(hiveCtx):

    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            query                                 string                     COMMENT   'query',
            pv                                    int                        COMMENT   'query周pv',
            terms                                 array<string>              COMMENT   'query分词',
            cid3                                  string                     COMMENT   'cid3',
            cid3_name                             string                     COMMENT   'cid3 名称',
            term_features                         map<string, array<float>>  COMMENT   '每个term对应的ctf, confidence, brand, icf, igm特征',
            ctf_icf                               array<string>              COMMENT   'ctf*icf计算出的term label分值',
            ctf_igm                               array<string>              COMMENT   'ctf*igm计算出的term label分值',
            ctf_confidence_icf                    array<string>              COMMENT   'ctf_confidence*icf 计算出的term label分值',
            ctf_confidence_igm                    array<string>              COMMENT   'ctf_confidence*igm 计算出的term label分值',
            ctf_brand_icf                         array<string>              COMMENT   '(ctf+brand*w)*icf 计算出的term label分值',
            ctf_brand_igm                         array<string>              COMMENT   '(ctf+brand*w)*igm 计算出的term label分值',
            ctf_confidence_brand_icf              array<string>              COMMENT   '(ctf_confidence+brand*w)*icf 计算出的term label分值',
            ctf_confidence_brand_igm              array<string>              COMMENT   '(ctf_confidence+brand*w)*igm 计算出的term label分值',
            click_weights                         array<string>              COMMENT   '基于用户行为生成的词权重分值',
            confidence_flag                       int                        COMMENT   '1: 用户行为产生的term排序与统计特征任何一种排序一致, 0: 用户行为产生的term排序与统计特征任何一种不一致',
            same_nums                             int                        COMMENT   '8种统计+1种行为相同的排序总数量[0,9]'
            )
        PARTITIONED BY(dt string, dim_type string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/{table_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(table_name=save_table)

    print(create_tbl)
    hiveCtx.sql(create_tbl)
  

def get_rank(weights):
    t_w = dict() 
    for x in weights:
        tw = x.split(':')
        if len(tw)!=2:
            continue
        t_w.setdefault(tw[0], float(tw[1]))
    # sort
    result = sorted(t_w.items(), key=lambda x : x[1], reverse=True)
    result = [x[0] for x in result]
    return ''.join(result)

def confidence_func(rows):
    
    for row in rows:
        rank_nums = dict()
        # 点击term排序
        click_rank = get_rank(row.avg_weight)
        rank_nums[click_rank] =1
        confidence_flag = 0
        # 统计term排序
        for i in range(8):

            f_rank = get_rank(row[i+6])
            # 计数相同的数量
            if f_rank not in rank_nums:
                rank_nums.setdefault(f_rank, 0)
            rank_nums[f_rank]+=1
            # 用户行为和统计的是否相同
            if click_rank == f_rank:
                confidence_flag = 1
        # 相同的最大数
        maxs = max(rank_nums.values())
       
       # return
        result = list()
        for i in range (15):
            result.append(row[i])
        result.append(confidence_flag)
        result.append(maxs)
        yield result 

def get_combine(hiveCtx):

    
    sql="""
        select b.query as query, pv, terms, cid3, cid3_name, term_features,  ctf_icf, ctf_igm, ctf_confidence_icf, ctf_confidence_igm, ctf_brand_icf, ctf_brand_igm,  ctf_confidence_brand_icf,  ctf_confidence_brand_igm, avg_weight
        from
        (
            select query, avg_weight
            from {click_table}
            where dt='2021-11-19' and dim_type='entropy'
        )a
        join
        (
            select query, pv, terms, cid3, cid3_name, term_features,  ctf_icf, ctf_igm, ctf_confidence_icf, ctf_confidence_igm, ctf_brand_icf, ctf_brand_igm,  ctf_confidence_brand_icf,  ctf_confidence_brand_igm 
            from {statistics_table}
            where dt='2021-11-20' and dim_type='entropy'
        )b
        on a.query=b.query
        """.format(click_table=click_table, statistics_table=statistics_table)

    print(sql)
    rdd = hiveCtx.sql(sql).rdd.cache()
    rdd.repartition(500).mapPartitions(lambda rows: confidence_func(rows))\
        .toDF(['query','pv', 'terms','cid3','cid3_name','term_features','ctf_icf','ctf_igm','ctf_confidence_icf','ctf_confidence_igm','ctf_brand_icf','ctf_brand_igm','ctf_confidence_brand_icf','ctf_confidence_brand_igm','click_weights','confidence_flag','same_nums']).registerTempTable("score_table")
    
    insert_sql = """
                insert overwrite table {save_table} partition(dt='{dt}', dim_type='{dim_type}')
                select * from score_table
                """.format(save_table=save_table, dt=dt_str, dim_type='entropy')

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
    yest_dt=dt + datetime.timedelta(-1)
    yest_str=yest_dt.strftime("%Y-%m-%d")

    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_combine(hiveCtx)


