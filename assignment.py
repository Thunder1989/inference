import numpy as np
import pandas as pd
import glob
import os
import sys
import pysax
import random
import rank_metrics

from time import time

from scipy.stats import rankdata
from scipy import bitwise_or

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ARI

from Bio import pairwise2

from scikits.statsmodels.tools.tools import ECDF

from matplotlib import pyplot as plt

types = ['AirFlowNormalized', 'AirValvePosition', 'DischargeAirTemp', 'HeatOutput', 'SpaceTemp', 'SpaceTempCoolingSetpointActive', 'SpaceTempHeatingSetpointActive']
type_mapping = {str(i):j for i,j in enumerate(types)}

def get_common_cols_files(building):
    equip = ['ahu','vav']
    for e in equip:
        find_common_cols(building, e)

def find_common_cols(building,equip):

    bid = building
    path = './ahu_property_file_'+bid+'_cut/'
    files = sorted(glob.glob(path+equip+'/*.csv'))
    df = pd.read_csv(files[0])
    col = df.columns.values
    common_col = col

    out_path = path+equip+'_common/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for f in files:
        df = pd.read_csv(f)
        col = df.columns.values
        if 'AirFlowNormalized' not in col:
            print f
        common_col = set(common_col) & set(col)

    for f in files:
        df = pd.read_csv(f)
        col = df.columns.values
        for c in col:
            if c not in common_col:
                df.drop(c, axis=1, inplace=True)

        df.to_csv(out_path + f.split('/')[-1], index=False)

def cut():
    files = sorted(glob.glob('./data/*.csv'))
    path = './data_ahu/'
    if not os.path.exists(path):
        os.mkdir(path)

    for f in files:
        df = pd.read_csv(f)
        df = df.iloc[:4*24*30, 10:] #take 1-month data, trim first few meta columns
        df = df.dropna(axis=1, how='all') #drop na columns
        df = df.select_dtypes(include=[np.number]) #drop nominal columns
        df = df.loc[:, (df != df.ix[0]).any()] #drop constant columns
        df.to_csv(path+f.split('/')[-1], index=False)
        print f, 'done'

def ts2str():
    str_input = []
    ts_type = []
    ts_cluster = []
    files = sorted(glob.glob('./data_cut/*.csv'))
    sax = pysax.SAXModel(window=8, stride=8, nbins=4, alphabet="ABCDE")
    for f, idx in zip(files, range(len(files))):
        df = pd.read_csv(f)
        #print len(df.columns)
        for col in df:
            rd = df[col]
            #plt.plot(rd)
            #plt.show()
            ts2str = sax.symbolize_signal(rd)
            str_input.append(''.join(ts2str))
            ts_type.append(col)
            ts_cluster.append(idx)

    #output = file('./ts2str_%s_%s_%s_%s'%(sax.window, sax.stride, sax.nbins, len(sax.alphabet)),'w')

    output = file('./ts2str','w')
    output.writelines(['%s\n'%seq for seq in str_input])
    output.close()

    le = LabelEncoder()
    label = le.fit_transform(ts_type)
    output = file('./ts_type','w')
    output.write('%s\n'%list(le.classes_))
    output.writelines(['%s\n'%l for l in label])
    output.close()

    output = file('./ts_cluster','w')
    output.writelines(['%s\n'%c for c in ts_cluster])
    output.close()

def str2sim():
    str_input = [i.strip() for i in open('./ts2str','r').readlines()]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                        min_df=2,
                                        analyzer='char',
                                        ngram_range=(2, 4))
    str2tfidf = tfidf_vectorizer.fit_transform(str_input)
    ts_sim = str2tfidf * str2tfidf.T
    #print ts_sim
    print tfidf_vectorizer.get_feature_names()
    np.savetxt('./str2sim.csv', ts_sim.toarray(), delimiter=',')
    #plt.imshow(ts_sim.toarray(), cmap=plt.cm.Blues)
    #plt.show()

def clustering():
    str_input = [i.strip() for i in open('./ts2str','r').readlines()]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                        min_df=2,
                                        analyzer='char',
                                        ngram_range=(2, 5))
    str2tfidf = tfidf_vectorizer.fit_transform(str_input)
    #print tfidf_vectorizer.get_feature_names()
    sc = SpectralClustering(n_clusters=113,
                            eigen_solver='arpack',
                            affinity="nearest_neighbors",
                            #assign_labels="discretize"
                            )
    y_pred = sc.fit_predict(str2tfidf)
    #y_true = [int(i.strip()) for i in open('./ts_cluster','r').readlines()]
    y_true = [i.strip() for i in open('./ts_type','r').readlines()]
    y_true = y_true[1:]
    print ARI(y_true, y_pred)
    y_shuffle = list(y_true)
    random.shuffle(y_shuffle)
    print ARI(y_true, y_shuffle)

def align():
    str_input = [i.strip() for i in open('./ts2str','r').readlines()]
    num = len(str_input)
    score = np.zeros((num, num))

    for i in range(num):
        for j in range(i):
            score[i][j] = pairwise2.align.globalxx(str_input[i],str_input[j],score_only=True)
    np.savetxt('./str2align.csv', score, delimiter=',')

def rank():
    str_input = [i.strip() for i in open('./ts2str','r').readlines()]
    cluster_label = np.asarray( [i.strip() for i in open('./ts_cluster','r').readlines()] )
    type_label = [i.strip() for i in open('./ts_type','r').readlines()]
    type_label = np.asarray( type_label[1:] )
    num = len(str_input)

    ap = []
    #num = 10
    acc_all = np.zeros((num,2))
    tp_all = np.zeros((num,2))
    t0 = time()
    ct = 0
    for i in range(num):
        if type_label[i] == '5' or type_label[i] == '6':
            ct += 1
        #    acc_all[i,:] = np.nan
        #    continue
        sim = np.zeros(num)
        cur = str_input[i]
        #print len(cur)
        for j in range(num):
            if j==i:
                #imply sim[j] = 0
                continue

            sim_raw = []
            k = 0
            win = 2*2
            stride = win/2
            tar = str_input[j]
            while k+win <= len(cur):
                tmp = pairwise2.align.globalms(cur[k:k+win], tar[k:k+win], 2,-1,-2,-.1, score_only=True)
                sim_raw.append(tmp)
                k += stride

            tmp = sum(sorted(sim_raw)[-10:])
            sim[j] = tmp

        rank = rankdata(sim, method='max')
        rank = len(rank) - rank #the function returns run in desceding order, reverse it

        nb_set = cluster_label[rank==0]
        #TBD: might want to check other ranking position, get unique rankings and iterate
        #also might need calculate FP
        acc_all[i,0] = int( cluster_label[i] in nb_set )
        tp_all[i,0] = sum( nb_set == cluster_label[i] ) / float(len(nb_set))

        '''
        idx = range(num)
        res = zip(idx, sim)
        res = sorted(res, key=lambda x: x[-1], reverse=True)

        rel = []
        idx, sim = zip(*res)

        res = []
        for k, r in enumerate(idx):
            rel.append( int(cluster_label[i] == cluster_label[r]) )
            #if cluster_label[i] == cluster_label[r]:
            #    res.append([ k, type_mapping[ type_label[r] ], sim[k] ])
            res.append([ k, cluster_label[r], sim[k] ])

        ap.append( rank_metrics.average_precision(rel) )

        print '----------------------------'
        print rel
        print 'query type:', type_mapping[type_label[i]], ', vav id', cluster_label[i]
        print res
        print rank_metrics.average_precision(rel)

        acc_all[i,0] = int(cluster_label[i] == cluster_label[idx[0]])
        acc_all[i,1] = int(cluster_label[i] == cluster_label[idx[1]])
        '''

        #raw_input('next')

    #print 'MAP:', np.mean(ap)
    #print 'count of pts', len(acc_all) - np.sum(np.isnan(top_acc), axis=0)
    print 'done in', time() - t0, 'seconds'
    print 'all top acc:', np.mean(acc_all, axis=0)
    print 'all top tp:', np.mean(tp_all, axis=0)
    #plot_cdf([acc_all, tp_all], 'all')
    df_out = pd.DataFrame([acc_all[:,0], tp_all[:,0]])
    df_out.T.to_csv('all_intra_stpt.csv', header=False, index=False)


    '''
    type_label = type_label[:num]
    acc_stpt = acc_all[bitwise_or(type_label=='5', type_label=='6'), :]
    tp_stpt = tp_all[bitwise_or(type_label=='5', type_label=='6'), :]
    print 'stpt top acc:', np.mean(acc_stpt, axis=0)
    print 'stpt top tp:', np.mean(tp_stpt, axis=0)
    #plot_cdf([acc_stpt, tp_stpt, fp_stpt], 'stpt_only')
    df_out = pd.DataFrame([acc_stpt[:,0], tp_stpt[:,0]])
    df_out.T.to_csv('stpt_only.csv', header=False, index=False)

    assert ct == len(acc_stpt)
    '''

def plot_cdf(arrays, fn):
    num = len(arrays)
    f, axarr = plt.subplots(num, sharex=True)
    for i, array in enumerate(arrays):
        array = array[:,0]
        cdf = ECDF(array)
        axarr[i].step(cdf.x, cdf.y, where='post', color='r')
    plt.savefig('%s.pdf'%fn, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    para = sys.argv[1]
    get_common_cols_files(para)
    #cut()
    #ts2str()
    #str2sim()
    #clustering()
    #align()
    #rank()

