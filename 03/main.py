import os
import matplotlib.pyplot as plt
from keras.utils import np_utils


from lab3_proto import *
from proto2 import *
from proto import mfcc, mspec_lab3
from prondict import prondict


np.random.seed(19890222)

def main():
    stateList = list(np.load("statelist.npy"))
    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

    # example = np.load('lab3_example.npz')['example'].item()
    # print(example.keys())
    # filename = 'train/man/nw/z43a.wav'
    # phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    # samples, sample = loadAudio(filename)
    # lmfcc = mfcc(samples)
    # wordTrans = list(path2info(filename)[2])
    # phoneTrans = words2phones(wordTrans, prondict)
    # viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

    # wordTrans = list(path2info(filename)[2])

    # utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    # # print(lmfcc.shape, example['lmfcc'].shape)
    # # print(viterbiStateTrans == example['viterbiStateTrans'])

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

    # x_train_mspec = X_train[0]['mspec']
    # for l in range(1, len(X_train)):
    #     print("{} / {}".format(l, len(X_train)))
    #     x_train_mspec = np.concatenate((x_train_mspec, X_train[l]['mspec']), axis=0)
    # np.save('x_train_mspec', x_train_mspec)
    # x_train_lmfcc = X_train[0]['lmfcc']
    # for l in range(1, len(X_train)):
    #     print("{} / {}".format(l, len(X_train)))
    #     x_train_lmfcc = np.concatenate((x_train_lmfcc, X_train[l]['lmfcc']), axis=0)
    # np.save('x_train_lmfcc', x_train_lmfcc)
    # x_validation_mspec = X_validation[0]['mspec']
    # for l in range(1, len(X_validation)):
    #     print("{} / {}".format(l, len(X_validation)))
    #     x_validation_mspec = np.concatenate((x_validation_mspec, X_validation[l]['mspec']), axis=0)
    # np.save('x_validation_mspec', x_validation_mspec)

    # x_validation_lmfcc = X_validation[0]['lmfcc']
    # for l in range(1, len(X_validation)):
    #     print("{} / {}".format(l, len(X_validation)))
    #     x_validation_lmfcc = np.concatenate((x_validation_lmfcc, X_validation[l]['lmfcc']), axis=0)
    # np.save('x_validation_lmfcc', x_validation_lmfcc)

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

    # x_train_lmfcc = np.load("x_train_lmfcc.npy").astype('float32')
    # x_train_mspec = np.load("x_train_mspec.npy").astype('float32')
    # x_validation_lmfcc = np.load("x_validation_lmfcc.npy").astype('float32')
    # x_validation_mspec = np.load("x_validation_mspec.npy").astype('float32')
    # x_test_lmfcc = np.load("x_test_lmfcc.npy").astype('float32')
    # x_test_mspec = np.load("x_test_mspec.npy").astype('float32')
    y_train_targets = np.load("y_train_targets.npy").astype('float32')
    y_validation_targets = np.load("y_validation_targets.npy").astype('float32')
    y_test_targets = np.load("y_test_targets.npy").astype('float32')

    # Dynamic

    # dynamic_x_train_mspec = np.load("dynamic_x_train_mspec.npy").astype('float32')
    # dynamic_x_validation_mspec = np.load("dynamic_x_validation_mspec.npy").astype('float32')
    # dynamic_x_test_mspec = np.load("dynamic_x_test_mspec.npy").astype('float32')

    dynamic_x_train_lmfcc = np.load("dynamic_x_train_lmfcc.npy").astype('float32')
    dynamic_x_validation_lmfcc = np.load("dynamic_x_validation_lmfcc.npy").astype('float32')
    dynamic_x_test_lmfcc = np.load("dynamic_x_test_lmfcc.npy").astype('float32')

    # X = {}
    # DX = [x_train_lmfcc, x_validation_lmfcc,
    #       x_test_lmfcc, x_train_mspec,
    #       x_validation_mspec, x_test_mspec]
    # names = ["x_train_lmfcc", "x_validation_lmfcc",
    #          "x_test_lmfcc", "x_train_mspec",
    #          "x_validation_mspec", "x_test_mspec"]
    # DY = [y_train_targets, y_validation_targets, y_test_targets]


    # b_idx = [
    #     [3, 2, 1, 0, 1, 2, 3],
    #     [2, 1, 0, 1, 2, 3, 4],
    #     [1, 0, 1, 2, 3, 4, 5],
    # ]
    # end_idx = [
    #     [7, 8, 9, 10, 11, 12, 11],
    #     [8, 9, 10, 11, 12, 11, 10],
    #     [9, 10, 11, 12, 11, 10, 9],
    # ]
    # end_idx = [
    #     [34, 35, 36, 37, 38, 39, 38],
    #     [35, 36, 37, 38, 39, 38, 37],
    #     [36, 37, 38, 39, 38, 37, 36],
    # ]

#    X[names[0]] = DX[0][0]
    # res = []
    # for name, utterance in zip(names[3:], DX[3:]):
    #     acc = 0
    #     res = []
    #     for feature in range(utterance.shape[1]):
    #         acc += utterance.shape[0]
    #         if feature > 36:
    #             res = np.concatenate((res, utterance[:, end_idx[feature-37]]), axis=1)
    #         elif feature < 3:
    #             if not len(res):
    #                 res = utterance[:, b_idx[feature]]
    #             else:
    #                 res = np.concatenate((res, utterance[:, b_idx[feature]]), axis=1)
    #         else:
    #             print(res.shape)
    #             res = np.concatenate((res, utterance[:, feature-3:feature+4]), axis=1)
    #     X[name] = np.array(res).reshape(len(res), utterance.shape[1]*7)
    #     np.save("dynamic_" + name, X[name])

    # # *************************************************************************

    # # ************************* NORMALIZE *************************************
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(dynamic_x_train_lmfcc)

    dynamic_x_train_lmfcc = scaler.fit_transform(dynamic_x_train_lmfcc,
                                                 scaler.get_params())
    dynamic_x_validation_lmfcc = scaler.fit_transform(dynamic_x_validation_lmfcc,
                                                      scaler.get_params())
    dynamic_x_test_lmfcc = scaler.fit_transform(dynamic_x_test_lmfcc,
                                                scaler.get_params())
    # dynamic_x_train_mspec = scaler.fit_transform(dynamic_x_train_mspec,
    #                                              scaler.get_params())
    # dynamic_x_validation_mspec = scaler.fit_transform(dynamic_x_validation_mspec,
    #                                                   scaler.get_params())
    # dynamic_x_test_mspec = scaler.fit_transform(dynamic_x_test_mspec,
    #                                             scaler.get_params())

    # # ***************************DATA PREP*************************************
    output_dim = len(stateList)
    y_train = np_utils.to_categorical(y_train_targets, output_dim)
    y_validation = np_utils.to_categorical(y_validation_targets, output_dim)
    y_test = np_utils.to_categorical(y_test_targets, output_dim)

    # # x_train = np.concatenate((x_train_lmfcc, x_train_mspec), axis=1)
    # # x_validation = np.concatenate((x_validation_lmfcc, x_validation_mspec), axis=1)
    # # x_test = np.concatenate((x_test_lmfcc, x_test_mspec), axis=1)

    # # *************************************************************************

    # # ************************* NEURAL NETWORK TRAINING ***********************
    from keras.models import Sequential
    from keras.layers import Dense

    # # ********************** dynamic lmfcc features ***********************************
    model = Sequential()
    model.add(Dense(256, input_dim=91, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(61, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(dynamic_x_train_lmfcc, y_train,
              batch_size=100,
              validation_data=(dynamic_x_validation_lmfcc, y_validation))

    # ********************** lmfcc features ***********************************
    # model = Sequential()
    # model.add(Dense(256, input_dim=13, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(61, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='sgd',
    #               metrics=['accuracy'])
    # model.fit(x_train_lmfcc, y_train,
    #           batch_size=256,
    #           validation_data=(x_validation_lmfcc, y_validation))

    # scores = model.predict(x_test_lmfcc)
    # print(np.sum(scores - y_test))

    # *************************************************************************
    # plt.plot(x_train_lmfcc[:86, 0], 'r')
    # plt.plot(scaler.transform(x_train_lmfcc[:86, 0]), 'b')
    # plt.show()

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
    # stateList = list(np.load("statelist.npy"))
    # phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

    # traindata = []
    # for root, dirs, files in os.walk('train'):
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
    #             traindata.append({'filename': fname, 'lmfcc': lmfcc,
    #                               'mspec': mspec, 'targets': targets})
    #         print("New file...")
    # np.savez('traindata.npz', traindata=traindata)
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
