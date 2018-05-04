import os
import matplotlib.pyplot as plt


from lab3_proto import *
from proto2 import *
from proto import mfcc, mspec_lab3
from prondict import prondict

def main():
    stateList = list(np.load("statelist.npy"))
    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    # filename = 'train/man/nw/z43a.wav'
    # phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    # samples, sample = loadAudio(filename)
    # lmfcc = mfcc(samples)
    # wordTrans = list(path2info(filename)[2])
    # phoneTrans = words2phones(wordTrans, prondict)
    # viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
    # targets = [stateList.index(state) for state in viterbiStateTrans]
    # print(targets)
    # X = np.load('traindata.npz')['traindata']
    # X_test = np.load('testdata.npz')['testdata']
    # # ************************ SPLIT DATA *************************************

    # validation_list = []
    # train_list = []
    # for d in X:
    #     if 'woman' in d['filename'] and d['filename'].endswith('b.wav'):
    #         validation_list.append(d)
    #     elif 'man' in d['filename'] and d['filename'].endswith('b.wav'):
    #         validation_list.append(d)
    #     else:
    #         train_list.append(d)
    # X_train = train_list
    # X_validation = validation_list

    # print(X_train[0]['lmfcc'].shape)
    # print("Concatenating data...")
    # y_train_targets = X_train[0]['targets']
    # for l in range(1, len(X_train)):
    #     print("{} / {}".format(l, len(X_train)))
    #     y_train_targets = np.concatenate((y_train_targets, X_train[l]['targets']), axis=0)
    # np.save('y_train_targets', y_train_targets)

    # y_validation_targets = X_validation[0]['targets']
    # for l in range(1, len(X_validation)):
    #     print("{} / {}".format(l, len(X_validation)))
    #     y_validation_targets = np.concatenate((y_validation_targets, X_validation[l]['targets']), axis=0)
    # np.save('y_validation_targets', y_validation_targets)

    # y_test_targets = X_test[0]['targets']
    # for l in range(1, len(X_test)):
    #     print("{} / {}".format(l, len(X_test)))
    #     y_test_targets = np.concatenate((y_test_targets, X_test[l]['targets']), axis=0)
    # np.save('y_test_targets', y_test_targets)

    # # x_validation_lmfcc = X_validation[0]['lmfcc']
    # # for l in range(1, len(X_validation)):
    # #     print("{} / {}".format(l, len(X_validation)))
    # #     x_validation_lmfcc = np.concatenate((x_validation_lmfcc, X_validation[l]['lmfcc']), axis=0)
    # # np.save('x_validation_lmfcc', x_validation_lmfcc)

    # x_test_mspec = X_test[0]['mspec']
    # for l in range(1, len(X_test)):
    #     print("{} / {}".format(l, len(X_test)))
    #     x_test_mspec = np.concatenate((x_test_mspec, X_test[l]['mspec']), axis=0)
    # np.save('x_test_mspec', x_test_mspec)
    # x_test_lmfcc = X_test[0]['lmfcc']
    # for l in range(1, len(X_test)):
    #     print("{} / {}".format(l, len(X_test)))
    #     x_test_lmfcc = np.concatenate((x_test_lmfcc, X_test[l]['lmfcc']), axis=0)
    # np.save('x_test_lmfcc', x_test_lmfcc)


    # lmfcc_train_x = np.array([d['lmfcc'] for d in X_train])

    # accum = 0
    # dim = 13
    # for utterance in lmfcc_train_x:
    #     accum += len(utterance)
    # lmfcc_train_x.reshape(accum, dim)
    # print(lmfcc_train_x.shape)
    # lmfcc_val_x = np.array([d['lmfcc'] for d in X_validation])
    # lmfcc_test_x = np.array([d['lmfcc'] for d in X_test])

    # mspec_train_x = np.array([d['mspec'] for d in X_train])
    # mspec_val_x = np.array([d['mspec'] for d in X_validation])
    # mspec_test_x = np.array([d['mspec'] for d in X_test])

    # train_y = np.array([d['targets'] for d in X_train])
    # val_y = np.array([d['targets'] for d in X_validation])
    # test_y = np.array([d['targets'] for d in X_test])

    x_train_lmfcc = np.load("x_data.npy").astype('float32')
    x_train_mspec = np.load("x_data_mspec.npy").astype('float32')
    x_validation_lmfcc = np.load("x_validation_lmfcc.npy").astype('float32')
    x_validation_mspec = np.load("x_validation_mspec.npy").astype('float32')
    x_test_lmfcc = np.load("x_test_lmfcc.npy").astype('float32')
    x_test_mspec = np.load("x_test_mspec.npy").astype('float32')
    y_train_targets = np.load("y_train_targets.npy").astype('float32')
    y_validation_targets = np.load("y_validation_targets.npy").astype('float32')
    y_test_targets = np.load("y_test_targets.npy").astype('float32')

    # *************************************************************************

    # ************************* NORMALIZE *************************************
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train_lmfcc)


    x_train_lmfcc()


    plt.plot(x_train_lmfcc[:86, 0], 'r')
    plt.plot(scaler.transform(x_train_lmfcc[:86, 0]), 'b')
    plt.show()

    # *************************************************************************

    # men = 0
    # women = 0
    # for d in X:
    #     if 'woman' in d['filename']:
    #         women += 1
    #     else:
    #         men += 1
    # print(men, women)
    # print(X[]['filename'])
    # y = np.load('testdata.npz')['testdata']

    # *******************Save to file *************************************
    # testdata = []
    # for root, dirs, files in os.walk('test'):
    #     for f in files:
    #         if f.endswith('.wav'):
    #             fname = os.path.join(root, f)
    #             samples, sample = loadAudio(fname)
    #             lmfcc = mfcc(samples)
    #             mspec = mspec_lab3(samples)
    #             wordTrans = list(path2info(fname)[2])
    #             phoneTrans = words2phones(wordTrans, prondict)
    #             viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
    #             targets = [stateList.index(state) for state in viterbiStateTrans]
    #             # Insert code for feature extraction
    #             testdata.append({'filename': fname, 'lmfcc': lmfcc,
    #                               'mspec': mspec, 'targets': targets})
    # np.savez('testdata.npz', testdata=testdata)
    # **********************************************************************


    # ex_samples = loadAudio('train/man/ae/z9z6531a.wav')
    # ex_info = path2info('train/man/ae/z9z6531a.wav')

    # phones = sorted(phoneHMMs.keys())
    # nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}


    # filename = 'train/man/nw/z43a.wav'
    # samples, samplingrate = loadAudio(filename)
    # lmfcc = mfcc(samples)

    # wordTrans = list(path2info(filename)[2])

    # phoneTrans = words2phones(wordTrans, prondict)
    # utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    # stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
    #               for stateid in range(nstates[phone])]

    # obsloglike = log_multivariate_normal_density_diag(lmfcc,
    #                                                   utteranceHMM['means'],
    #                                                   utteranceHMM['covars'])


    # viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

    # frames2trans(viterbiStateTrans, outfilename='z43a.lab')

    # print(viterbiStateTrans)
    # print(vi_mat[1])
    # print()
    # print(stateTrans)
    # print()
    # print(stateList)

    ## print(stateTrans)
    # print(utteranceHMM['transmat'])

    # stateList = [ph + '_' + str(id) for ph in phones
    #              for id in range(nstates[ph])]
    # np.save("statelist", stateList)




if __name__ == '__main__':
    main()
