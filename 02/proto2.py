import numpy as np
from tools2 import *
from prondict import *
import sklearn.mixture as sm
import matplotlib.pylab as plt

def concatHMMs(hmm_models, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    M = len(namelist)*len(namelist)
    trans_mat = np.zeros((M+1, M+1))
    i, j = 1, 1
    for idx, phoneme in enumerate(namelist):
        trans_mat[idx*3:i*3+1, idx*3:j*3+1] = hmm_models[phoneme]['transmat']
        j += 1
        i += 1
#     plt.pcolormesh(trans_mat)
#     plt.axis([0,10,10,0])
#     plt.colorbar()
#     plt.show()
    
    means = hmm_models[namelist[0]]['means']
    for p in namelist[1:]:
        means = np.concatenate((means,hmm_models[p]['means']))
    
    covars = hmm_models[namelist[0]]['covars']
    for p in namelist[1:]:
        covars = np.concatenate((covars,hmm_models[p]['covars']))
    startprob = np.zeros((10,))
    startprob[0] = 1
    return {'transmat':trans_mat, 'means':means, 'covars':covars, 'startprob':startprob}


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(emlik, startprob, transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = emlike.shape
    forward_prob = np.zeros((N,M))
    startprob.dot(emlike)


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """

def plot_obsloglike(obsloglike):
    plt.figure()
    plt.pcolormesh(obsloglike.T)
#     plt.xlabel('features')
#     plt.ylabel('features')
    plt.colorbar()


def main():
    data = np.load('lab2_data.npz')['data']
    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    example = np.load('lab2_example.npz')['example'].item()
    modellist = {}
    for digit in prondict.keys():
        modellist[digit] = ['sil'] + prondict[digit] + ['sil']
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, modellist['o'])
    
    # CHECK FOR CORRECTNESS.  same as example['obsloglik']
    obsloglike = sm.log_multivariate_normal_density(example['lmfcc'],
                                            wordHMMs['o']['means'],
                                            wordHMMs['o']['covars'],
                                            'diag')
    x = sm.log_multivariate_normal_density(data[22]['lmfcc'],
                                            wordHMMs['o']['means'],
                                            wordHMMs['o']['covars'],
                                            'diag')
#     plot_obsloglike(x)
#     plot_obsloglike(obsloglike)
#     plt.show()
    forward(example['obsloglik'], np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat'])) 
    return 0


if __name__=='__main__':
    main()

