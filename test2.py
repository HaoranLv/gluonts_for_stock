import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
def load_json(filename):
    data = []
    with open(filename, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            datai = json.loads(line)
            data.append(datai)
    return data
def parse_data_ar(dataset):
    data = []
    start = pd.Timestamp("2020-07-01 17:00:00")
    begin = int((start - pd.Timestamp("2020-01-02 00:00:00")) / pd.Timedelta("30T"))
    for t in dataset:
        datai = {'target': [t['target'][3][begin:],t['target'][8][begin:]], 'start': t['start']}
        if 'id' in t:
            datai['id'] = t['id']
        if 'cat' in t:
            datai['cat'] = t['cat']
        if 'dynamic_feat' in t:
            datai['FieldName.FEAT_DYNAMIC_REAL'] = t['dynamic_feat']
        data.append(datai)
    return data
N = 60+10
x = np.linspace(0, 10, N)
i=1
ones = np.ones(10)
mean=np.load('./gpvar_mean.npy')
g01=np.load('./gpvar_01.npy')
g09=np.load('./gpvar_09.npy')
gt=np.load('./gpvar_gt.npy')
# total=load_json('./glts_total_multi_tar.json')
# total=parse_data_ar(total)
# y=total[i]['target'][1][:70]
a=np.load('./gpvar.npy')
fig, ax = plt.subplots()
xbuy_mean=mean[i,1,:10]
x01=g01[i,0,:10]
x09=g09[i,0,:10]
gtt=gt[i,0,:10]
ax.plot(x[-10:], xbuy_mean, color='blue') # Pl
ax.fill_between(x[-10:], x01, x09, color='blue', alpha=0.5)
ax.plot(x[-10:], gtt, color='red') # Plot the original signal
# ax.plot(x[:], y, color='red')
#
plt.show()