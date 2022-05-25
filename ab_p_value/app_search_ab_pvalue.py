#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==================================================
# @Created   : 2019/8/19 15:08
# @Author    : caimengqian <caimengqian@jd.com>
# @Objective : TODO
# @File      : sp_app_search_ab_keyword_dim_pvalue_top_cidx.py
# 使用到的表:
# app.app_idata_bdl_search_query_log    -- 曝光
# app.app_sdl_yinliu_search_click_log   -- 点击
# app.app_sdl_yinliu_search_order_log   -- 订单
#
# 口径:
# app渠道、L2去作弊、去广告、下单口径
# ==================================================
import sys
import argparse
import os
import time
import datetime
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext


reload(sys)
sys.setdefaultencoding('utf8')

#SUGGEST_DBPATH = os.environ['SUGGEST_DBPATH']
SUGGEST_DBPATH = 'hdfs://ns1013/user/recsys/suggest/app.db'
keyword_table = "app_ab_mtest_keyword_szda"
pvalue_table = "app_ab_mtest_pvalue_szda"
base_table = "app_ab_mtest_base_szda_v2"


def init_sp():
    global conf, sc, hiveCtx
    conf = SparkConf()
    conf.set('spark.sql.codegen', 'true')
    conf.set('spark.rdd.compress', 'true')
    conf.set('spark.broadcast.compress', 'true')
    sc = SparkContext(conf=conf, appName=(os.path.basename(sys.argv[0]) + args.date_e))
    sc.setLogLevel('error')
    hiveCtx = HiveContext(sc)
    hiveCtx.setConf('hive.auto.convert.join', 'true')
    hiveCtx.setConf('hive.exec.dynamic.partition.mode', 'nonstrict')
    hiveCtx.setConf('hive.exec.dynamic.partition', 'true')
    hiveCtx.setConf('spark.sql.shuffle.partitions', '2000')
    hiveCtx.setConf('spark.default.parallelism', '2000')
    hiveCtx.setConf("hive.exec.max.dynamic.partitions", "1000")
    hiveCtx.setConf("spark.sql.autoBroadcastJoinThreshold", "200000")
    hiveCtx.setConf("spark.shuffle.consolidateFiles", "true")
    hiveCtx.setConf("mapreduce.job.reduces", "1000")
    hiveCtx.setConf("spark.shuffle.compress", "true")
    # 合并小文件
    # hiveCtx.setConf("spark.sql.hive.mergeFiles", "true")
    hiveCtx.sql("use app")


def create_table():
    create_pvalue_table_sql = """
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name}(
            indicator_type      string comment '指标',
            base_variance       double comment '方差', 
            base_mean           double comment '均值', 
            base_simlpe_size    double comment '样本量',
            test_variance       double comment '方差', 
            test_mean           double comment '均值', 
            test_simlpe_size    double comment '样本量'
        )
    PARTITIONED BY (dt string, dim_type string)
    STORED AS ORC
    LOCATION '{SUGGEST_DBPATH}/{table_name}'
    TBLPROPERTIES (
        'orc.compress'='SNAPPY',
        'SENSITIVE_TABLE'='FALSE')
    """.format(SUGGEST_DBPATH=SUGGEST_DBPATH, table_name=pvalue_table)
    print("create_pvalue_table_sql:" + create_pvalue_table_sql)
    hiveCtx.sql(create_pvalue_table_sql)

    create_base_table_sql = """
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name}(
            version         string comment 'base/test',
            pv              bigint, 
            uv              bigint, 
            click           bigint,
            uclick          bigint,
            orderlines      bigint,
            gmv             bigint,
            uv_value        double,
            ucvr            double,
            uctr             double,
            ctr             double  
        )
    PARTITIONED BY (dt string, dim_type string)
    STORED AS ORC
    LOCATION '{SUGGEST_DBPATH}/{table_name}'
    TBLPROPERTIES (
        'orc.compress'='SNAPPY',
        'SENSITIVE_TABLE'='FALSE')
    """.format(SUGGEST_DBPATH=SUGGEST_DBPATH, table_name=base_table)
    print("create_base_table_sql:" + create_base_table_sql)
    hiveCtx.sql(create_base_table_sql)

    # 搜索表
    create_search_table_sql = """
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name}(
            version    string comment 'ab compare',
            uuid    string comment 'uuid',
            date    string comment 'date'
        )
    PARTITIONED BY (dt string)
    STORED AS ORC
    LOCATION '{SUGGEST_DBPATH}/{table_name}'
    TBLPROPERTIES (
        'orc.compress'='SNAPPY',
        'SENSITIVE_TABLE'='FALSE')
    """.format(SUGGEST_DBPATH=SUGGEST_DBPATH, table_name='search_wxmina_ab_test_table')
    print("search_table:" + create_search_table_sql)
    hiveCtx.sql(create_search_table_sql)

    # click表
    create_click_table_sql = """
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name}(
            version    string comment 'ab compare',
            uuid    string comment 'uuid',
            logid    string comment 'logid',
            date    string comment 'date'
        )
    PARTITIONED BY (dt string)
    STORED AS ORC
    LOCATION '{SUGGEST_DBPATH}/{table_name}'
    TBLPROPERTIES (
        'orc.compress'='SNAPPY',
        'SENSITIVE_TABLE'='FALSE')
    """.format(SUGGEST_DBPATH=SUGGEST_DBPATH, table_name='click_wxmina_ab_test_table')
    print("click_table:" + create_click_table_sql)
    hiveCtx.sql(create_click_table_sql)

    # order 表
    create_order_table_sql = """
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name}(
            version    string comment 'ab compare',
            uuid    string comment 'uuid',
            order_id    string comment 'order_id',
            sku    string comment 'sku',
            after_prefr_amount_1    float comment 'after_prefr_amount_1',
            user_log_acct    string comment 'user_log_acct',
            date    string comment 'date'
        )
    PARTITIONED BY (dt string)
    STORED AS ORC
    LOCATION '{SUGGEST_DBPATH}/{table_name}'
    TBLPROPERTIES (
        'orc.compress'='SNAPPY',
        'SENSITIVE_TABLE'='FALSE')
    """.format(SUGGEST_DBPATH=SUGGEST_DBPATH, table_name='order_wxmina_ab_test_table')
    print("order_table:" + create_order_table_sql)
    hiveCtx.sql(create_order_table_sql)


def pvalue_func(dt_s, dt_e, base, test, dim):
    # ==================== 获取触发词 ====================
    keyword_sql = """
        select
            dt, trim(lower(keyword)) as keyword
        from
            {keyword_table}
        where
            dt >= '{dt_s}' and dt <= '{dt_e}' and dim_type = '{dim}'
        group by dt, trim(lower(keyword))
    """.format(dt_s=dt_s, dt_e=dt_e, keyword_table=keyword_table, dim=dim)
    print("keyword_sql:" + keyword_sql)
    #hiveCtx.sql(keyword_sql).registerTempTable("keyword_table")

    # ==================== 读取初始表 ====================
    # 商品曝光
    hiveCtx.sql("""
        select ext_columns['wq_unionid'] as uuid, dt
                from app.wx_qq_idata_bdl_search_exposure_log 
                where dt >= '{dt_s}' and dt <= '{dt_e}'
                and chan_type=5
    """.format(dt_s=dt_s, dt_e=dt_e)).registerTempTable("query_mid")
    #hiveCtx.cacheTable("query_mid")

    # 搜索
    hiveCtx.sql("""
            select version, uuid, dt
            from 
            (
                    select case when regexp_extract(ext_columns['drop_ver'], '{test}', 0) != '' then 'test'
                                when regexp_extract(ext_columns['drop_ver'], '{base}', 0) != '' then 'base'
                            end as version,
                        ext_columns['wq_unionid'] as uuid,dt
                        from app.app_idata_bdl_all_click_jx_log
                        where dt >= '{dt_s}' and dt<='{dt_e}'
                        and biz_type = 'search_query'
                        and dim_type in ('wxmina')
                        and cheat_tag['L2_new']=0
                        and (regexp_extract(ext_columns['drop_ver'], '{base}',0) != '' or
                        regexp_extract(ext_columns['drop_ver'], '{test}',0) != '')
            )a
            """.format(dt_s=dt_s, dt_e=dt_e, test='SAK7_SM_WT_L19723', base='SAK7_SM_WT_L19722')).registerTempTable("search_mid")
  
    # join
    hiveCtx.sql("""
            select version, b.uuid as uuid, b.dt as dt 
            from
            (
            select version, uuid, dt
            from search_mid
            group by version, uuid, dt
            )a
            join
            (
                select uuid, dt
                from query_mid
            )b
            on a.uuid=b.uuid and a.dt=b.dt
        """).registerTempTable("search_v2_mid")

    #insert
    insert_sql = """
        insert overwrite table {table_name} partition(dt='{dt}')
        select * from search_mid""".format(table_name='search_wxmina_ab_test_table', dt='2022-05-18')

    print("insert_sql:" + insert_sql)
    #hiveCtx.sql(insert_sql)

    # 点击
    hiveCtx.sql("""
        select version, b.uuid as uuid, logid, b.dt as dt
        from
        (
           select version, uuid,dt
           from search_v2_mid
           group by version, uuid,dt
        )a
        join
        (
            select other_ext_colums['wq_unionid'] uuid, logid, dt
            from app.app_sdl_yinliu_search_click_log
            where dt >= '{dt_s}' and dt <= '{dt_e}'
            and dim_type = 'wxmina' and source='0'
            and cheat_tag['L2_new'] = 0
        )b
        on a.uuid=b.uuid and a.dt=b.dt

    """.format(dt_s=dt_s, dt_e=dt_e)).registerTempTable("click_mid")
    
    hiveCtx.cacheTable("click_mid")
    
    #insert
    insert_sql = """
        insert overwrite table {table_name} partition(dt='{dt}')
        select * from click_mid """.format(table_name='click_wxmina_ab_test_table', dt='2022-05-18')

    print("insert_sql:" + insert_sql)
    #hiveCtx.sql(insert_sql)

    # 订单
    hiveCtx.sql("""
        select version, b.uuid as uuid, order_id, sku, after_prefr_amount_1, user_log_acct, b.dt as dt
        from
            (
                select version, uuid,dt
                from search_v2_mid
                group by version, uuid, dt
            )a
            join
            (
                select pvid, other_ext_colums['wq_unionid'] uuid, order_id, sku, after_prefr_amount_1, user_log_acct, dt
            from
                app.app_sdl_yinliu_search_order_log
            where 
                dt >= '{dt_s}' and dt <= '{dt_e}' 
                and dim_type = 'wxmina'
                and out_tag = 1 --出库口径
                and cheat_tag['L2_new'] = 0
                and key_word  is not null and trim(key_word) != ''
                and dim_ab_tag = 1
            )b
            on a.uuid=b.uuid and a.dt=b.dt
    """.format(dt_s=dt_s, dt_e=dt_e)).registerTempTable("order_mid")
    hiveCtx.cacheTable("order_mid")
    
    #insert
    insert_sql = """
        insert overwrite table {table_name} partition(dt='{dt}')
        select * from order_mid """.format(table_name='order_wxmina_ab_test_table', dt='2022-05-18')

    print("insert_sql:" + insert_sql)
    #hiveCtx.sql(insert_sql)

    # =================== 基础数据计算 ====================
    # 曝光
    query_sql = """
        select version, count(1) as pv, count(distinct uuid) as uv
        from search_mid
        group by version
    """.format(test=test, base=base)
    print("query_sql:" + query_sql)
    hiveCtx.sql(query_sql).registerTempTable("query_v2_table")
    hiveCtx.cacheTable("query_v2_table")
   
    #曝光2
    query_sql = """
        select version, count(1) as pv, count(distinct uuid) as uv
        from search_v2_mid
        group by version
    """.format(test=test, base=base)
    print("query_sql:" + query_sql)
    hiveCtx.sql(query_sql).registerTempTable("query_table")
    hiveCtx.cacheTable("query_table")

    # 点击
    click_sql = """
        select version, count(1) as pv, count(distinct uuid) as uv
        from click_mid
        group by version
    """.format(test=test, base=base)
    print("click_sql:" + click_sql)
    hiveCtx.sql(click_sql).registerTempTable("click_table")
    hiveCtx.cacheTable("click_table")

    # 订单
    order_sql = """
        select version, count(distinct concat(order_id, '_', sku)) as orderlines, sum(after_prefr_amount_1) as gmv
        from order_mid
        group by version
    """.format(test=test, base=base)
    print("order_sql:" + order_sql)
    hiveCtx.sql(order_sql).registerTempTable("order_table")
    hiveCtx.cacheTable("order_table")

    # ================ 方差, 均值, 样本量计算 ================
    # ctr
    # 平方和
    ctr_square_sql = """
        select version, sum(power(click, 2)) as click_power_sum, sum(click) as all_click
        from 
            (select version, logid, count(1) as click
             from click_mid
             group by logid, version)tt
        group by version
    """.format(test=test, base=base)
    print("ctr_square_sql:" + ctr_square_sql)
    hiveCtx.sql(ctr_square_sql).registerTempTable("ctr_square_table")
    # 方差, 均值, 样本量计算
    ctr_sql = """
        select 'ctr' as indicator_type, tb1.version, 
               tb2.click_power_sum/(tb1.pv+0.00) - power(tb2.all_click/(tb1.pv+0.00), 2) as variance,
               tb2.all_click/(tb1.pv+0.00) as mean,
               tb1.pv as simlpe_size
        from 
            query_table as tb1
        left join 
            ctr_square_table as tb2
        on tb1.version=tb2.version
    """
    print("ctr_sql:" + ctr_sql)
    hiveCtx.sql(ctr_sql).registerTempTable("ctr_table")

    # cvr
    # 平方和
    cvr_square_sql = """
            select version, sum(power(orderlines, 2)) as orderlines_power_sum, sum(orderlines) as all_orderlines
            from 
                (select version, user_log_acct, count(distinct concat(order_id, '_', sku)) as orderlines
                 from order_mid
                 group by user_log_acct, version)tt
            group by version
        """.format(test=test, base=base)
    print("cvr_square_sql:" + cvr_square_sql)
    hiveCtx.sql(cvr_square_sql).registerTempTable("cvr_square_table")
    # 方差, 均值, 样本量计算
    uctr_sql = """
        select 'uctr' as indicator_type, tb1.version, 
               tb2.click_power_sum/(tb1.uv+0.00) - power(tb2.all_click/(tb1.uv+0.00), 2) as variance,
               tb2.all_click/(tb1.uv+0.00) as mean,
               tb1.uv as simlpe_size
        from 
            query_table as tb1
        left join 
            ctr_square_table as tb2
        on tb1.version=tb2.version
    """
    print("uctr_sql:" + uctr_sql)
    hiveCtx.sql(uctr_sql).registerTempTable("uctr_table")

    # ucvr
    # 方差, 均值, 样本量计算
    ucvr_sql = """
        select 'ucvr' as indicator_type, tb1.version, 
               tb2.orderlines_power_sum/(tb1.uv+0.00) - power(tb2.all_orderlines/(tb1.uv+0.00), 2) as variance,
               tb2.all_orderlines/(tb1.uv+0.00) as mean,
               tb1.uv as simlpe_size
        from 
            query_table as tb1
        left join 
            cvr_square_table as tb2
        on tb1.version=tb2.version
    """
    print("ucvr_sql:" + ucvr_sql)
    hiveCtx.sql(ucvr_sql).registerTempTable("ucvr_table")

    # uv_value
    # 平方和
    uv_value_square_sql = """
            select version, sum(power(gmv, 2)) as gmv_power_sum, sum(gmv) as all_gmv
            from 
                (select version, user_log_acct, sum(after_prefr_amount_1) as gmv
                 from order_mid
                 group by user_log_acct, version)tt
            group by version
        """.format(test=test, base=base)
    print("uv_value_square_sql:" + uv_value_square_sql)
    hiveCtx.sql(uv_value_square_sql).registerTempTable("uv_value_square_table")
    # 方差, 均值, 样本量计算
    uv_value_sql = """
        select 'uv_value' as indicator_type, tb1.version, 
               tb2.gmv_power_sum/(tb1.uv+0.00) - power(tb2.all_gmv/(tb1.uv+0.00), 2) as variance,
               tb2.all_gmv/(tb1.uv+0.00) as mean,
               tb1.uv as simlpe_size
        from 
            query_table as tb1
        left join 
            uv_value_square_table as tb2
        on tb1.version=tb2.version
    """
    print("uv_value_sql:" + uv_value_sql)
    hiveCtx.sql(uv_value_sql).registerTempTable("uv_value_table")

    # ===================== 结果数据 ======================
    union_sql = """
        select indicator_type, version, variance, mean, simlpe_size
        from ctr_table
        union
        select indicator_type, version, variance, mean, simlpe_size
        from uctr_table
        union
        select indicator_type, version, variance, mean, simlpe_size
        from ucvr_table
        union
        select indicator_type, version, variance, mean, simlpe_size
        from uv_value_table
    """
    hiveCtx.sql(union_sql).registerTempTable("union_table")

    result_sql = """
        select tb1.indicator_type,
               base_variance, base_mean, base_simlpe_size,
               test_variance, test_mean, test_simlpe_size
        from
            (select indicator_type, variance as base_variance, mean as base_mean, simlpe_size as base_simlpe_size
             from union_table
             where version = 'base')tb1
        join
            (select indicator_type, variance as test_variance, mean as test_mean, simlpe_size as test_simlpe_size
             from union_table
             where version = 'test')tb2
        on tb1.indicator_type=tb2.indicator_type
    """
    print("result_sql:" + result_sql)
    hiveCtx.sql(result_sql).registerTempTable("result_table")

    insert_sql = """
        insert overwrite table {table_name} partition(dt='{dt}', dim_type='{dim}')
        select * from result_table""".format(table_name=pvalue_table, dt=dt_e, dim=dim)
    print("insert_sql:" + insert_sql)
    hiveCtx.sql(insert_sql)


def base_func(dt_e, dim):
    base_sql = """
    select version, pv, uv, click, uclick, orderlines, gmv,
           gmv/cast(uv as double) AS uv_value, 
           cast(orderlines as double)/cast(uv as double) AS ucvr,
           cast(uclick as double)/cast(uv as double) AS uctr,
           cast(click as double)/cast(pv as double) AS ctr
    from 
         (select tb1.version, tb1.pv, tb1.uv, 
                 tb2.pv as click, tb2.uv as uclick,  
                 tb3.orderlines, tb3.gmv
          from query_table as tb1
          join click_table as tb2
          on tb1.version = tb2.version
          join order_table as tb3
          on tb2.version = tb3.version)
    """
    print("base_sql:" + base_sql)
    hiveCtx.sql(base_sql).registerTempTable("base_table")

    insert_sql = """
    insert overwrite table {table_name} partition (dt='{dt}', dim_type='{dim}')
    select version, pv, uv, click, uclick, orderlines, gmv,
           uv_value, ucvr, uctr, ctr
    from base_table
    """.format(table_name=base_table, dt=dt_e, dim=dim)
    print("insert_sql:" + insert_sql)
    hiveCtx.sql(insert_sql)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date_s", help="work_date", default='2022-04-23')
    parser.add_argument("--date_e", help="work_date", default='2022-05-04')
    parser.add_argument("--dim", help="keyword dim", default='result_all_v6') #result_all, 2022-04-23~2022-05-04
    parser.add_argument("--base", help="base group/msg['mtest']", default='SAK7_SM_WT_L19722')
    parser.add_argument("--test", help="test group/msg['mtest']", default='SAK7_SM_WT_L19723')

    args = parser.parse_args()
    print("%s parameters: %s" % (sys.argv[0], str(args)))
    print("%s begin at %s" % (sys.argv[0], str(datetime.datetime.now())))
    begin_time = time.time()

    init_sp()

    create_table()
    print("create table done!")
    pvalue_func(args.date_s, args.date_e, args.base, args.test, args.dim)
    print("pvalue done!")
    base_func(args.date_e, args.dim)
    print("base info done!")
    print("ALL Job Done!!")

    end_time = time.time()
    print("%s end at %s" % (sys.argv[0], str(datetime.datetime.now())))
    print("%s total cost time:%s" % (sys.argv[0], str(end_time - begin_time)))
