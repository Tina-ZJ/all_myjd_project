import argparse
import io
import numpy as np
import pandas as pd
import sys
from scipy import stats
from numpy import math


def ztest_t_pvalue(base, test):
    # base="1467290.0124270378,257.7721372968972,19141.0"
    # test="1594532.3079061382,355.4767952840318,1866.0#19141.3079061382,1467290.4767952840318,1866.0"
    try:
        s_base = float(base.split(',')[0])
        x_base = float(base.split(',')[1])
        n_base = float(base.split(',')[2])
    except:
        print("Error:Please input correct arguments!")

    try:
        s_test = float(test.split(',')[0])
        x_test = float(test.split(',')[1])
        n_test = float(test.split(',')[2])
    except:
        print("Error:Please input correct arguments!")

    try:
        z_value = (x_test - x_base)/np.math.sqrt((s_test/n_test + s_base/n_base))
        df = round(pow((s_test/n_test+s_base/n_base), 2) / (pow((s_test/n_test), 2)/(n_test-1) +
                                                            pow((s_base/n_base), 2)/(n_base-1)))
        
        p = min(1, 2 * (1 - stats.t.cdf(abs(z_value), df)))
        if math.isnan(p) is True:
            return ''
        else:
            # return str(round(p, 4))
            return str(p)
    except:
        return ''


def diff_func(a, b):
    base = float(a)
    test = float(b)
    return str(round((test/base-1)*100, 4)) + '%'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_i")
    parser.add_argument("--file_v")

    args = parser.parse_args()

    result = [[], [], [], []]
    with io.open(args.file_i, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            items = line.strip().split()
            if items[0] == 'base':
                result_idx = 0
            elif items[0] == 'test':
                result_idx = 1
            else:
                continue
            result[result_idx] = items
    # compute diff
    if len(result[0]) != 0 and len(result[0]) != 0:
        result[2] = ['diff'] + ['']*6 + [diff_func(result[0][7], result[1][7]),
                                         diff_func(result[0][8], result[1][8]),
                                         diff_func(result[0][9], result[1][9]),
                                         diff_func(result[0][10], result[1][10])]
    else:
        print("base info error!")
        sys.exit(1)

    variance_dict = dict()
    with io.open(args.file_v, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            items = line.strip().split()
            variance_dict[items[0]] = [','.join(items[1:4]), ','.join(items[4:7])]

    # compute p value
    result[3] = [u'p value'] + [''] * 6
    indicator_list = ['uv_value', 'ucvr', 'cvr', 'ctr']
    for indicator in indicator_list:
        if indicator in variance_dict:
            b = variance_dict[indicator][0]
            t = variance_dict[indicator][1]
            result[3].append(ztest_t_pvalue(b, t))
        else:
            print("variance info error!")
            sys.exit(1)

    # write result
    df = pd.DataFrame(result, columns=['version', 'pv', 'uv', 'click', 'uclick', 'orderlines', 'gmv', 'uv_value', 'ucvr', 'cvr', 'ctr'])
    df.to_csv("result.csv", index=False)




