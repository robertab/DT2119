import os

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
    X = np.load('train.npz')['train']
    y = np.load('testdata.npz')['testdata']

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
