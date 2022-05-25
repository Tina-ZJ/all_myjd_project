import re
import numpy as np
from BILSTM_CRF import BILSTM_CRF

# a = np.arange(1,5)
# b = a.reshape([2,2])
# print b
#
# b1 = np.expand_dims(np.array([3,4]), 0)
# b2 = b1 + b
# print b2
#
#
# c = np.expand_dims(b, 0)
# d = np.expand_dims(b, 1)
# e = c + d
# print "c: "
# print c
# print "d: "
# print d
# print "e: "
# print e

model = BILSTM_CRF(100, 5, 10)