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
    # SET LAST ELEMENT TO 1.
    trans_mat[-1,-1] = 1

    # PLOT MATRIX
#     plt.pcolormesh(trans_mat)
#     plt.axis([0,10,10,0])
#     plt.colorbar()
#     plt.show()

    # CONCAT MEAN MATRICES
    means = hmm_models[namelist[0]]['means']
    for p in namelist[1:]:
        means = np.concatenate((means,hmm_models[p]['means']))
    # CONCAT COVAR MARTICES    
    covars = hmm_models[namelist[0]]['covars']
    for p in namelist[1:]:
        covars = np.concatenate((covars,hmm_models[p]['covars']))
    # SET STARTPROB VECTOR
    startprob = np.zeros((10,1))
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

def forward(emlike, startprob, transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
        alpha (forward_prob) is a joint probability of observations AND state at time step t.
    """
    N, M = emlike.shape
    forward_prob = np.zeros((N,M))
    # INITIALISATION
    for i in range(transmat.shape[1]-1):
        forward_prob[i,0] = emlike[i,0] + startprob[i,0]
    # FORWARD ALGORITHM
    for t in range(1,M):
        for i in range(N):
            forward_prob[i,t] = logsumexp(forward_prob[:,t-1] + transmat[:-1,i]) + emlike[i,t] # skip last row in transmatrix since it is absorbing. 
    return forward_prob


def backward(emlike, startprob, transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
        beta (backward_prob) is a conditional probability of the observations GIVEN the state at time step t.  
    """
    N, M = emlike.shape
    backward_prob = np.zeros((N,M))
    # INIT
    backward_prob[:,-1] = 0
    # BACKWARD PASS
    for t in range(M-2,-1,-1):
        for i in range(N):
            backward_prob[i,t] = logsumexp(transmat[:-1, i] + backward_prob[:,t+1] + emlike[i,t+1])
    return backward_prob

def viterbi(emlike, startprob, transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N,M = emlike.shape
    vi = np.zeros((N,M))    
    path = np.zeros((N,M))
    obsseq = np.zeros((M,))
    # INIT
    vi[:,0] = emlike[:,0] + startprob[:-1,0]
    for t in range(1,M):
        for i in range(N):
            # MAX PROB OF PREVIOUS VI-timestep * P(we go from each of prevois states to i) * P(we obeserve i at timestep t). (use + insted of *, since log domain)
            vi[i,t] = np.max(vi[:,t-1] + transmat[:-1,i]) + emlike[i,t]
            path[i,t] = np.argmax(vi[:,t-1] + transmat[:-1,i])
    zt = np.argmax(vi[:,-1])
    obsseq[M-1] = zt
    for t in range(M-1,0,-1):
        zt = path[int(zt),t]
        obsseq[t-1] = zt
    return vi, obsseq


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
    wordHMMs = {}
    for digit in prondict.keys():
        modellist[digit] = ['sil'] + prondict[digit] + ['sil']
        wordHMMs[digit] = concatHMMs(phoneHMMs, modellist[digit])
    
    # CHECK FOR CORRECTNESS.  same as example['obsloglik']
    obsloglike = sm.log_multivariate_normal_density(example['lmfcc'],
                                            wordHMMs['o']['means'],
                                            wordHMMs['o']['covars'],
                                            'diag')
    alpha_mat = forward(example['obsloglik'].T, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat'])) 
#     print(alpha_mat.T - example['logalpha'])
    # LOGLIKELIHOOD
    loglike = logsumexp(alpha_mat.T[-1,:].T)
    vi, obsseq = viterbi(example['obsloglik'].T, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat'])) 
#     print(example['vloglik'][1] - obsseq)
    beta_mat = backward(example['obsloglik'].T, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat'])) 
    print(np.round(beta_mat.T - example['logbeta']))
    return 0


if __name__=='__main__':
    main()

