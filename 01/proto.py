import numpy as np
import matplotlib.pyplot as plt
from more_itertools import windowed
from scipy.signal import *
from scipy.fftpack import fft
from tools import trfbank
from scipy.fftpack.realtransforms import dct
from tools import lifter
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
    something = None 
    N, D = x.shape[0], 13
    M, D = y.shape[0], 13
    # d = dist(x, y) / len(x) + len(y)
    LD = np.zeros((N, M))
    AD = np.zeros((N, M))
    path = []
    # Initialize the dynamic programming algorithm
    for i, x_frame in enumerate(x):
        for j, y_frame in enumerate(y):
            LD[i, j] = dist(x_frame, y_frame)

    AD[0, :], AD[:, 0] = LD[0, :], LD[:, 0]
    for i in range(1, AD.shape[0]):

        for j in range(1, AD.shape[1]):
            AD[i, j] = min(AD[i, j-1], AD[i-1, j], AD[i-1, j-1])

            
def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))
        
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
    cmap = plt.get_cmap('jet')
    plt.pcolormesh(cov_mat, cmap=cmap)
    plt.title('Correlation Matrix')
    plt.xlabel('features')
    plt.ylabel('features')
    plt.axis([0,12,12,0])
    plt.colorbar()    
    plt.show()



def main():
    cmap = plt.get_cmap('jet')
    # print("Hello, world!")
    example = np.load('data/lab1_example.npz')['example'].item()

    # Data contains array of dictionaries
    data = np.load('data/lab1_data.npz')['data']

    ex1 = data[0]['samples']
    ex2 = data[0]['samples']

    dtw(ex1, ex2, euclidean)
    

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
    
