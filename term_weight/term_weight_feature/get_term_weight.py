# -*-coding=utf-8
import sys
import math
import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os

source_table='sz_algo_cid3_words_statistics_features'
save_table='sz_algo_terms_weights_rank'


def create_table(hiveCtx):

    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            term                                  string             COMMENT   'query分词',
            cid3                                  string             COMMENT   'cid3',
            cid3_name                             string             COMMENT   'cid3 名称',
            ctf_icf                               float              COMMENT   'ctf*icf计算出的term label分值',
            ctf_igm                               float              COMMENT   'ctf*igm计算出的term label分值',
            ctf_confidence_icf                    float              COMMENT   'ctf_confidence*icf 计算出的term label分值',
            ctf_confidence_igm                    float              COMMENT   'ctf_confidence*igm 计算出的term label分值',
            ctf_brand_icf                         float              COMMENT   '(ctf+brand*w)*icf 计算出的term label分值',
            ctf_brand_igm                         float              COMMENT   '(ctf+brand*w)*igm 计算出的term label分值',
            ctf_confidence_brand_icf              float              COMMENT   '(ctf_confidence+brand*w)*icf 计算出的term label分值',
            ctf_confidence_brand_igm              float              COMMENT   '(ctf_confidence+brand*w)*igm 计算出的term label分值',
            rank_icf                              int                COMMENT   'rank ctf_brand_icf',
            rank_igm                              int                COMMENT   'rank ctf_brand_igm'
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
        
        cid3, cid3_name, term, brand, ctf, confidence, icf, igm = row
        if term.strip()=='':
            continue
        brand = float(brand)
        igm=100*igm
        # combine ways
        ctf_icf = ctf*icf
        ctf_igm = ctf*igm
        ctf_confi_icf = confidence*icf
        ctf_confi_igm = confidence*igm
        ctf_brand_icf = (ctf + brand*0.2)*icf
        ctf_brand_igm = (ctf + brand*0.02)*igm
        ctf_confi_brand_icf = (confidence + brand*0.2)*icf
        ctf_confi_brand_igm = (confidence + brand*0.01)*igm

        yield term, cid3, cid3_name, ctf_icf, ctf_igm, ctf_confi_icf, ctf_confi_igm, ctf_brand_icf, ctf_brand_igm, ctf_confi_brand_icf, ctf_confi_brand_igm


def get_weights(hiveCtx):

    

    sql="""
            select cid3, cid3_name, term, brand_flag, ctf, ctf_confidence, icf, igm
            from {source_table}
            where dt='{dt}' and dim_type='{dim_type}'
        """.format(source_table=source_table, dt=dt_str, dim_type=dim_type)
    

    print(sql)
    rdd = hiveCtx.sql(sql).rdd.cache()
    rdd.repartition(500).mapPartitions(lambda rows: score_func(rows))\
        .toDF(['term','cid3','cid3_name','ctf_icf','ctf_igm','ctf_confidence_icf','ctf_confidence_igm','ctf_brand_icf','ctf_brand_igm','ctf_confidence_brand_icf','ctf_confidence_brand_igm']).registerTempTable("score_table")
   
    # sort ctf_brand_icf, ctf_brand_igm
    sql = """
            select a.term as term, a.cid3 as cid3, rank_icf, rank_igm
            from
             (
                select term, cid3, ctf_brand_icf, row_number() over(partition by cid3 order by ctf_brand_icf desc) as rank_icf
                from score_table
             )a
            join
            (
                select term, cid3, ctf_brand_igm, row_number() over(partition by cid3 order by ctf_brand_igm desc) as rank_igm
                from score_table
            )b
            on a.term=b.term and a.cid3=b.cid3
        """
    print(sql)
    hiveCtx.sql(sql).registerTempTable("rank_table")

    # combine
    sql="""
        select b.term as term, b.cid3 as cid3, cid3_name, ctf_icf, ctf_igm, ctf_confidence_icf, ctf_confidence_igm, ctf_brand_icf, ctf_brand_igm, ctf_confidence_brand_icf, ctf_confidence_brand_igm, rank_icf, rank_igm
        from
        (
            select term, cid3, rank_icf, rank_igm
            from rank_table
        )a
        join
        (
            select term, cid3, cid3_name, ctf_icf, ctf_igm, ctf_confidence_icf, ctf_confidence_igm, ctf_brand_icf, ctf_brand_igm, ctf_confidence_brand_icf, ctf_confidence_brand_igm
            from score_table
        )b
        on a.term=b.term and a.cid3=b.cid3
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
    yest_dt=dt + datetime.timedelta(-1)
    yest_str=yest_dt.strftime("%Y-%m-%d")
    dim_type='base'
    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_weights(hiveCtx)


