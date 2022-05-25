# -*- coding:utf-8 -*-
import os,sys
import six
import time
import shutil
import json

debug = False


# debug = True
def loge(s):
    print(s)

def logw(s):
    print(s)

def logi(s):
    print(s)

def logd(s):
    if debug:
        print(s)
    pass

######################################################################################################
#################                      configure load                     ############################
######################################################################################################
def read_cfg_2(path, dmap, kkey, vkey, kidx, vidx, ktype=str, vtype=str, head=False):
    i = 0
    for l in open(path):
        if isinstance(l, str) and six.PY2:
            l = l.decode('utf-8')
        a = l[:-1].split('\t')
        i += 1
        if i == 1 and head:
            for i in range(len(a)):
                if a[i] == kkey:
                    kidx = i
                elif a[i] == vkey:
                    vidx = i

            continue
        if ktype != str:
            k = ktype(a[kidx])
        else:
            k = a[kidx]

        if vtype == list:
            v = a[vidx].split(',')
        elif vtype == str:
            v = a[vidx]
        else:
            v = vtype(a[vidx])
        # dmap[ktype(a[kidx])] = vtype(a[vidx])
        dmap[k] = v


def read_cfg(path, dmap, kkey="key", vkey="value", ktype=str, vtype=str):
    read_cfg_2(path, dmap, kkey, vkey, kidx=-1, vidx=-1, ktype=ktype, vtype=vtype, head=True)


def read_cfg_i(path, dmap, kidx=0, vidx=1, ktype=str, vtype=str, head=False):
    read_cfg_2(path, dmap, kkey="", vkey="", kidx=kidx, vidx=vidx, ktype=ktype, vtype=vtype, head=head)


def get_term_ids(terms, term_ids_dict, max_seq_len=-1, padding=False, ignore_space=True):
    innids = list()
    words = list()
    if isinstance(terms, str) and six.PY2:
        terms = terms.decode('utf-8')
    for t in terms:
        if isinstance(t, str) and six.PY2:
            t = t.decode('utf-8')
        if ignore_space and t == ' ':
            continue
        innids.append(term_ids_dict.get(t, term_ids_dict['[UNK]']))
        words.append(t)

        if max_seq_len > 0 and len(innids) >= max_seq_len:
            break

    if padding and max_seq_len > 0:
        for i in range(len(innids), max_seq_len):
            innids.append(term_ids_dict['[PAD]'])
    return innids, words


######################################################################################################
#################                      model postprocess                  ############################
######################################################################################################
# cates [[id, w], [id, w]]
# filter cate_weight_min and cdf
def postprocess_cid(indexs, weights, outtype='normal', cate_weight_min=0.02):
    cates = []
    max_w, max_w_i = 0 , 0
    for i, w in zip(indexs, weights):
        if w > cate_weight_min:
            cates.append([i, w])
        if w > max_w:
            max_w = w
            max_w_i = i
    def list_2_dict(cates):
        r = {}
        r['cids'] = [c[0] for c in cates]
        r['cid_weights'] = [c[1] for c in cates]
        # for compatibility
        r['last_class'] = [[c[0], 'unk', c[1]] for c in cates]
        return r


    cates = sorted(cates, key=lambda x:x[1], reverse=True)
    if outtype == 'offcache': return list_2_dict(cates)

    if len(cates) > 16: cates = cates[:16]

    def calc_cate_cdf(cids):
        ### calc cdf
        total_score = 0
        cdf_score = 0
        for c,w in cids:
            total_score += w
        if total_score == 0:
            total_score = 0.1

        cate_l = list()
        for c,w in cids:
            t = w/total_score
            if cdf_score<0.9 or w >= 0.1:
                cdf_score += t
                cate_l.append([c,w])
        return cate_l

    cates = calc_cate_cdf(cates)

    if len(cates) == 0:
        logw('cates empty %f'%(max_w))
        cates.append([max_w_i, max_w])
    return list_2_dict(cates)

def predict_error_ret(info):
    return {'status': 'ERROR', 'info': info}

######################################################################################################
#################                      spark utils                        ############################
######################################################################################################
def get_model_path_in_spark(spark_app):
    d = json.load(open('qptools_model_info.json'))
    if spark_app not in  d.keys() or 'fname' not in d[spark_app].keys():
        return "ERROR not key %s in %s"%(spark_app, d), ''
    model_path = d[spark_app]['fname']
    if sys.version_info.major == 0 and isinstance(model_path, unicode):
        model_path = model_path.encode('utf-8')
    return 'OK', model_path

def init_spark_env_3(model_path):
    #os.system('unzip %s.zip')
    import zipfile
    with zipfile.ZipFile(model_path) as zf:
        zf.extractall()
    return 'OK'


def init_spark_env_2(model_name):
    var_list = ['variables.data-00000-of-00001', 'variables.index']
    vdir = model_name + '/variables'

    if not os.path.exists(vdir):
        try:
            os.makedirs(vdir)
        except:
            return 'may makedirs by others %s' % vdir

    for v in var_list:
        src = model_name + v
        dest = vdir + '/' + v
        if os.path.exists(src):
            try:
                shutil.move(src, dest)
            except Exception as e:
                # logw('may mv by others %s'%src)
                # return 'may mv by others %s'%src
                return 'can not mv %s : %s' % (src, e)
        else:
            if os.path.exists(dest):
                continue
            return 'not exit src %s' % (src)

    for v in var_list:
        if not os.path.exists(vdir + '/' + v):
            return 'not exit variables  %s/%s' % (vdir, v)

    for root, dirs, files in os.walk("./"):
        if root != './':
            continue
        for f in files:
            if f.startswith(model_name):
                try:
                    shutil.move(f, '%s/%s' % (model_name, f[len(model_name):]))
                except Exception as e:
                    logw('may mv by others')
                    return 'mv error  %s : %s' % (f, e)

    return 'OK'


def init_spark_env(model_path):
    unlock_file = '__unlock__' + model_path
    lock_file = '__lock__' + model_path
    finish_file = '__finish__' + model_path
    ok_file = '__OK__' + model_path
    err_file = '__ERROR__' + model_path

    def wait_unlock():
        for _ in range(100):
            if os.path.exists(finish_file):
                if os.path.exists(ok_file):
                    return 'OK'
                if os.path.exists(err_file):
                    return 'ERROR'
            logw('init wait_unlock sleep 0.5')
            time.sleep(0.5)
        return 'ERROR wait timeout %s'%model_path

    import random
    st = random.randint(100, 400) * 0.01
    logw('init sleep st %f' % st)
    time.sleep(st)
    if not os.path.exists(unlock_file) and  not os.path.exists(lock_file):
        return 'ERORR not exists %s and not exists %s'%(unlock_file, lock_file)
    if os.path.exists(unlock_file):
        try:
            shutil.move(unlock_file, lock_file)
        except:
            return wait_unlock()
    else:
        return wait_unlock()

    ret = init_spark_env_3(model_path)
    os.system('touch ' + finish_file)
    if ret == 'OK':
        os.system('touch ' + ok_file)
    else:
        os.system('touch ' + err_file)
    return ret





######################################################################################################
#################                      aiflow utils                        ############################
######################################################################################################
def dump_yaml(ymlpath, model_name, version_id, output_str):
    d = dict()
    d['mode'] = 'check_score_offline_qp'
    d['arch'] = '9n'
    d['batch_size'] = 1
    d['eps'] = 1e-4
    d['host'] = '10.180.239.246'
    d['port'] = 8810
    model_root = '/export/App/tensorflows/stage/'
    d['model_root'] = model_root
    d['model_name'] = model_name
    d['version_id'] = int(version_id)
    d['input_type'] = 'raw'
    d['output_str'] = output_str
    d['test_file'] = 'assets.extra/test_4_proto'
    d['test_res_file'] = 'assets.extra/predict_result'
    d['feature_stats_json'] = 'assets.extra/feature_stats.json'
    d['new_arch'] = True
    import yaml
    # 写入到yaml文件
    with open(ymlpath, "w", encoding="utf-8") as f:
        yaml.dump(d, f)


def os_system(cmd, check=False):
    if check:
        s = input("是否执行 %s. (Y/N)" % cmd)
        if s != "Y":
            os._exit(-1)
    logi('execute : %s' % cmd)
    os.system(cmd)


def gen_model_info(model_pred, model_dir, verify_input_path, max_test_count=1000):
    # model_pred.init(model_dir)
    mfeatures = model_pred.get_feature_des()

    input_key = mfeatures['input']
    other_input_key = []
    if 'other_input' in mfeatures.keys():
        other_input_key = mfeatures['other_input']

    verify_infos = []
    for l in open(verify_input_path):
        t = json.loads(l)
        query = t['inputs'][input_key]
        other_feature = []
        for k in other_input_key:
            if not isinstance(t[k], list):
                other_feature.append(t[k])
            else:
                other_feature.append(t[k])
        if len(other_feature) == 0:
            input_tensor_features = model_pred.get_feature_ids(query)
        else:
            input_tensor_features = model_pred.get_feature_ids(query, other_feature)

        rd = model_pred.tf_predict(input_tensor_features)
        # d = model_pred.process_result(j)

        r = dict()
        for f in input_tensor_features:
            r[f.name] = f.values
            # logd('features %s'%v.name)
        ret = dict()
        ret['features'] = r
        ret['raw_result'] = {}
        for k in mfeatures['output_tensor']:
            ret['raw_result'][k] = rd[k]
        verify_infos.append(ret)
        if len(verify_infos) >= max_test_count: break
    return verify_infos


def gen_aiflow_package(model_pred, out_dir, model_name, verify_input_path):
    model_dir = model_pred.model_dir
    version_id = model_pred.version
    logi('version_id %s'%version_id)

    # remove model file
    # org_model_dir, model_name,version_id = args.model_dir, args.model_name, version_id
    out_dir = out_dir + '/' + version_id
    if os.path.exists(out_dir):
        os_system('rm -rf %s' % out_dir, False)
    os_system("mkdir -p %s" % out_dir)

    assets_dir = "%s/assets.extra" % (out_dir)
    if not os.path.exists(assets_dir):
        os.mkdir(assets_dir)

    '''
    test_4_proto(测试数据文件)：tsv格式数据
    predict_result(验分基准数据文件)：tsv格式数据
    test_config.yml(验分脚本配置文件)：yaml格式
    feature_stats.json(离线特征元数据文件)
    '''
    fd = dict()
    mfeatures = model_pred.get_feature_des()

    for a in mfeatures['input_tensor']:
        f = dict()
        f["feature_key"] = a[0]
        f["data_type"] = a[1]
        if len(a) >= 3:
            f['column_type'] = a[2]
        else:
            f["column_type"] = 'variable_length'
        if f['data_type'] == 'string':
            f["default_value"] = ''
        elif f['data_type'] == 'float':
            f["default_value"] = 1.0
        else:
            f["default_value"] = 1
        fd[a[0]] = f
    json.dump(fd, open(assets_dir + "/feature_stats.json", 'w'), indent=4)

    ###### 验证文件数据
    test_4_proto_f = open(assets_dir + "/test_4_proto", 'w')
    predict_result_f = open(assets_dir + "/predict_result", 'w')
    output_str = ''
    is_head = True
    count = 0

    verify_infos = gen_model_info(model_pred, model_dir, verify_input_path)
    for d in verify_infos:
        # dump test_4_proto head
        if is_head:
            test_4_proto_f.write("%s\n" % '\t'.join(d['features'].keys()))
            predict_result_f.write("%s\n" % '\t'.join(d['raw_result'].keys()))
            output_str = ','.join(d['raw_result'].keys())
            is_head = False
        a = []
        for k, v in d['features'].items():
            a.append(','.join([str(i) for i in v]))
        test_4_proto_f.write("%s\n" % ('\t'.join(a)))
        a = []
        for k, v in d['raw_result'].items():
            a.append(','.join([str(i) for i in v]))
        predict_result_f.write("%s\n" % ('\t'.join(a)))

    test_4_proto_f.close()
    predict_result_f.close()

    dump_yaml(assets_dir + '/test_config.yml', model_name + '_raw', version_id, output_str)
    logd('gen 9n model : %s' % (assets_dir))

    os_system("cp %s/saved_model.pbtxt %s/saved_model.pbtxt " % (model_dir, out_dir))
    os_system("md5sum %s/saved_model.pbtxt > %s/status" % (model_dir, out_dir))
    os_system("echo '%s' > %s/version" % (version_id, out_dir))
    os_system("cp -r %s/variables %s/variables" % (model_dir, out_dir))
    if os.path.exists(model_dir+'/assets'):
        os_system("cp -r %s/assets %s/assets " % (model_dir, out_dir))
    # os_system('cp -r %s/assets.extra %s/assets.extra'%( model_dir, out_dir))
    logi('gen 9n model : %s' % (out_dir))


######################################################################################################
#################                      misc                               ############################
######################################################################################################
import os

default_models_dir = os.environ['HOME'] + '/.qpmodels'


def download_model(model_name, model_url='', rootdir=default_models_dir):
    model_url_dict = {
        'hanmodel_v1': 'http://storage.jd.local/query-parser/QPModel/han/release/hanmodel_20210125.zip?Expires=3770882973&AccessKey=TA75Nt4TQ4RX7cLc&Signature=rygxhz%2FAQ6NV%2BBDRXCQK%2Bsu62V4%3D',
        "t2q_model_latest": "http://storage.jd.local/query-parser/QPModel/t2q/t2q_model_latest.zip?Expires=3757991237&AccessKey=TA75Nt4TQ4RX7cLc&Signature=9BNramzSBSI41B4kVfipPCCiQVc%3D",
        "mhanmodel_latest": "http://storage.jd.local/query-parser/QPModel/mhan/mhanmodel_latest.zip?Expires=3754578651&AccessKey=TA75Nt4TQ4RX7cLc&Signature=DdxJFOnZl%2FAJ8ODw9re3kbjh%2FuQ%3D"
    }

    if model_url == '':
        model_url = model_url_dict[model_name]

    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    if os.path.exists(os.path.join(rootdir, 'hanmodel_v1')):
        os.system('rm -rf ' + os.path.join(rootdir, 'hanmodel_v1'))

    cmd = "cd {rootdir}; wget \"{modle_url}\" -O {rootdir}/hanmodel_v1.zip; unzip hanmodel_v1.zip; rm -r hanmodel_v1.zip".format(
        rootdir, rootdir, rootdir)
    logd(cmd)
    os.system(cmd)


