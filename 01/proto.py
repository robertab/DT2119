import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm,mlab
from more_itertools import windowed
from scipy.signal import *
from scipy.fftpack import fft
from tools import trfbank
from scipy.fftpack.realtransforms import dct
from tools import lifter, todigit2labels
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.mixture import GaussianMixture
import sys
np.set_printoptions(threshold=np.nan)
np.random.seed(19860330)
# import more_itertools.windowed

# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    return np.asarray(list(windowed(samples, winlen, step=winshift, fillvalue=0)))


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    A = [1]
    B = [1, -p]
    return lfilter(B,A, input, axis=1)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    # The amount of samples in one frame
    frame_length = 400
    return input * hamming(frame_length, sym=0)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    ps = np.abs(fft(input, nfft))**2
    return ps


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    NFFT = input.shape[1]
    return np.log(input.dot(trfbank(samplingrate, NFFT).T))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return dct(input, type=2, axis=1, norm='ortho')[:,:nceps]

def plot_features(data):
    # NOW DO FOR ALL DATA
    plt.figure(1)
    plt.title('lifted cosine transform')
    c = 1
    for d in data:
        if d['digit']=='4':
            print(d['gender'])
            plt.subplot(410+c)
            mfcc_data = mfcc(d['samples'])
            plt.pcolormesh(mfcc_data.T, cmap=cmap)
#             plt.plot(mfcc_data)
            print(mfcc_data.shape)
            c+=1
    plt.figure(2)
    c = 1
    j= 1
    for d in data:
        if d['digit']=='4':
            print(d['gender'])
            plt.subplot(410+c)
            mfcc_data = mfcc(d['samples'])
            print(mfcc_data.shape)
            plt.plot(mfcc_data[:,7])
            c+=1
    plt.show()

def concatenate_data(data):
    utterances = np.empty((0, 13))
    for d in data:
        utterances = np.concatenate((utterances, mfcc(d['samples'])), axis=0)
    return utterances

def correlations(data, sampling_rate=20000, lift=False):
    utterances = np.empty((0,13))
    for d in data:
        if not lift:
            frames = enframe(d['samples'], 400, 200)
            preemph = preemp(frames, 0.97)
            windowed = windowing(preemph)
            spec = powerSpectrum(windowed, 512)
            mspec = logMelSpectrum(spec, sampling_rate)
            ceps = cepstrum(mspec, 13)
            utterances = np.concatenate((utterances, mspec),axis=0)
        else:
            utterances = np.concatenate((utterances, mfcc(d['samples'])),axis=0)
    return np.corrcoef(utterances.T)

def plot_cov(cov_mat):
    print(cov_mat.shape)
    cmap = plt.get_cmap('jet')
    plt.pcolormesh(cov_mat, cmap=cmap)
    plt.title('Correlation Matrix')
    plt.xlabel('features')
    plt.ylabel('features')
    plt.axis([0,12,12,0])
    plt.colorbar()
    plt.show()

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lengths of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N = x.shape[0]
    M = y.shape[0]
    global_dist = 0
    LD = np.zeros((N, M))
    AD = np.zeros((N, M))
    path_mat = []
    # Initialize the dynamic programming algorithm
    # Start out by filling a matrix of local distance values
    # between each frame.
    for i, x_frame in enumerate(x):
        for j, y_frame in enumerate(y):
            LD[i, j] = dist(x_frame, y_frame)

    nrows, ncols = AD.shape
    # set first row
    for row in range(1,nrows):
        AD[row,0] = AD[row-1,0]+LD[row,0]
    # set first col
    for col in range(1,ncols):
        AD[0,col] = AD[0,col-1] + LD[0,col]

    for row in range(1, nrows): # Start from 1 to avoid out of bounds
        for col in range(1, ncols):
            minimum_dist = LD[row, col] + min(AD[row, col-1],   # To the left
                                              AD[row-1, col],   # Above
                                              AD[row-1, col-1]) # The diagonal
            AD[row, col] = minimum_dist

    backtracking = True
    i, j = nrows-1, ncols-1
    path_mat.append((i,j))
    while backtracking:
        min_dist =  min(AD[i, j-1],
                        AD[i-1, j],
                        AD[i-1, j-1])
        min_idx = np.where(AD==min_dist)
        i, j = min_idx[0][0], min_idx[1][0]
        path_mat.append((i,j))
        if i == 0 and j == 0:
            backtracking = False

    global_dist = AD[nrows-1,ncols-1] / (len(x) + len(y))
    return LD, AD, path_mat, global_dist

def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))

def get_global_dist(data):
    N = data.shape[0]
    GD = np.zeros((N,N))
    row, col = GD.shape
    for i in range(row):
        x = mfcc(data[i]['samples'])
        for j in range(col):
            y = mfcc(data[j]['samples'])
            _, _, _, GD[i,j] = dtw(x,y,euclidean)
    return GD

def select_number(data, digit):
    X_test = np.empty((0, 13))
    c = 0
    for d in data:
        if d['digit'] == digit and d['speaker'] == 'bm' and d['repetition'] == 'a' and d['gender'] == 'man':
            print(d['digit'], d['speaker'], d['repetition'], d['gender'])
            X_test = np.concatenate((X_test, mfcc(d['samples'])), axis=0)
            c+=1
#         elif d['digit'] == digit and d['speaker'] == 'ew' and d['repetition'] == 'b':
#             print(d['digit'], d['speaker'], d['repetition'], d['gender'])
#             X_test = np.concatenate((X_test, mfcc(d['samples'])), axis=0)

    return X_test

def main():
    cmap = plt.get_cmap('jet')
    # print("Hello, world!")
    example = np.load('data/lab1_example.npz')['example'].item()

    # Data contains array of dictionaries
    data = np.load('data/lab1_data.npz')['data']

    # ***** GAUSSIAN MIXTURE MODEL *****

    # Get test data for one digit
    test_data_7 = select_number(data, '7')
#     test_data = np.concatenate((test_data,select_number(data, '9')))
    # Train data on all utterances
    mfcc_mat = concatenate_data(data)

    components = [4,8,16,32]
    for c in components:
        # Create model
        gmm = GaussianMixture(c, covariance_type='diag',max_iter=1000, verbose=1)
        # Train on all data
        gmm.fit(mfcc_mat)
        # Predict on #7
        y_7 = gmm.predict_proba(test_data_7)
        y_idx_7 = gmm.predict(test_data_7)
        print(y_7.shape)
        print(len(y_idx_7))
        print('pred idx_7:' + str(y_idx_7))
        cov = gmm.covariances_
        means = gmm.means_
        
        import scipy.stats
        x_axis=np.arange(-200,200,1)
#         fig ,ax = plt.subplots(c,13)
        for i in range(c):
            for j in range(means.shape[1]):
                plt.plot(scipy.stats.norm.pdf(x_axis,means[i,j],cov[i,j]))
#                 plt.plot(scipy.stats.norm.pdf(x_axis,mfcc_mat.mean(axis=1),mfcc_mat.var(axis=1)))
            plt.show() 
        print(test_data_7.shape)

#         color=cm.rainbow(np.linspace(0,1,c))
#         for i,c_ in zip(range(c),color):
#             plt.plot(y_7[:,i],c=c_, label = 'component: '+str(i))
# #             plt.plot(y_7,c=c_, label = 'component: '+str(i))
#             plt.legend()

#         plt.title('Gaussian mixture, Posterior probabilities')
#         plt.xlabel('components')
#         plt.ylabel('frame')
#         plt.pcolormesh(y_7, cmap=cmap)
# #         a = gmm.weights_.reshape(2,int(c/2))
# #         plt.plot(gmm.weights_)
# #         print(a.shape)
# #         plt.pcolormesh(a,cmap = cmap)
#         plt.colorbar()
#         plt.show()

#     plt.gca()
#     plt.scatter(mfcc_mat[:, 0], mfcc_mat[:, 1], mfcc_mat[:, 2], c=colors)
#                                                 mfcc_mat[:, 3],
#                                                 mfcc_mat[:, 4],
#                                                 mfcc_mat[:, 5],
#                                                 mfcc_mat[:, 6],
#                                                 mfcc_mat[:, 7],
#                                                 mfcc_mat[:, 8],
#                                                 mfcc_mat[:, 9],
#                                                 mfcc_mat[:, 10],
#                                                 mfcc_mat[:, 11],
#                                                 mfcc_mat[:, 12]),
#                                                 c=colors)
    
############# USE BELOW FOR GETTING THE PLOTS NEEEDED FOR REPORT ##############
    # **********************************

    # test_data = []
    # for d in data:
    #     if d['digit'] == '7':
    #         print(d['gender'])
    #         test_data.append((mfcc(d['samples']), d['gender'], d['digit']))


    # LD, AD, path, global_dist = dtw(test_data[0][0], test_data[2][0], euclidean)
    # x = [x[0] for x in path]
    # y = [x[1] for x in path]
    # plt.title("Best path for minimum distortion between two utterances")
    # plt.xlabel("Utterance for digit {} spoken by {}".format(str(test_data[0][2]), test_data[0][1]))
    # plt.ylabel("Utterance for digit {} spoken by {}".format(str(test_data[2][2]), test_data[2][1]))
    # plt.plot(x, y, c='r')
    # # plt.pcolormesh(AD, cmap=cmap)
    # plt.show()



    # ex1 = mfcc(data[0]['samples'])
    # ex2 = mfcc(data[1]['samples'])
    # ex3 = mfcc(data[2]['samples'])
    # ex4 = mfcc(data[3]['samples'])

    # LD, AD, path, d1 = dtw(ex1,ex2,euclidean)
    # x = [x[0] for x in path]
    # y = [x[1] for x in path]
    # plt.plot(x,y) # plot the path in the Accumulated distance matrix
    # plt.pcolormesh(AD, cmap=cmap)
    # plt.show()
    #
    # labels = todigit2labels(data)
    # print(labels)

    # GD = get_global_dist(data)
    # Hierarchy clustering. Clusering based on MAX distance
    # clusters = linkage(GD,method='complete')
    #
    # dendrogram(clusters, labels=labels)
    # plt.show()
    # dendrogram(clusters, leaf_label_func=tidigit2labels(data))


    # TEST FOR CORRECT CALCULATIONS
    # pe = preemp(frames)
    # win = windowing(pe)
    # power_spec = powerSpectrum(win, 512)
    # lms = logMelSpectrum(power_spec, 20000)
    # mfcc = cepstrum(lms,13)

    # frames = enframe(data, 400, 200)
    # pe = preemp(frames)
    # print(pe.shape)
    # win = windowing(pe)
    # power_spec = powerSpectrum(win, 512)
    # lms = logMelSpectrum(power_spec, 20000)
    # lift = False
    # cov_mat = correlations(data, 20000, lift)
    # plot_cov(cov_mat)


#     lmfcc_data = mfcc(data)
#     print(lmfcc_data.shape)


#     plt.figure(1)
#     plt.subplot(211)
#     plt.title('discrete cosine transform')
#     plt.pcolormesh(mfcc.T, cmap=cmap)
#     plt.subplot(212)
#     plt.title('Sinusodial lifting of the cosine transform')
#     plt.pcolormesh(lmfcc.T, cmap=cmap)

    # plt.show()
    # print(len(example['samples']))
    # print(np.sum(frames - example['frames']))
    ## print(frames[1])
    # print(example['frames'][0])
#     plt.plot(pe)
#     plt.figure()

#     plt.pcolormesh(pe)

    # hm = windowing(pe)




if __name__ == '__main__':
    main()

