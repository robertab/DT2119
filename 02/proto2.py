import numpy as np
from tools2 import *
from prondict import *
import sklearn.mixture as sm
import matplotlib.pylab as plt
np.set_printoptions(threshold=np.nan)
np.seterr(divide='ignore')
np.random.seed(400)

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
#     M = len(namelist)*len(namelist)
    M = len(namelist)
    trans_mat = np.zeros((M*3+1, M*3+1))
#     print(trans_mat.shape)
    i, j = 1, 1
    for idx, phoneme in enumerate(namelist):
#         print(hmm_models[phoneme]['transmat'].shape)
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
    startprob = np.zeros((M*3+1,1))
    startprob[0] = 1
    return {'transmat':trans_mat, 'means':means, 'covars':covars, 'startprob':startprob}


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
    for i in range(N):
        forward_prob[i,0] = emlike[i,0] + startprob[i,0]
    # FORWARD ALGORITHM
    for t in range(1,M):
        for i in range(N):
            forward_prob[i, t] = logsumexp(forward_prob[:, t-1] + transmat[:-1, i]) + emlike[i, t] # skip last row in transmatrix since it is absorbing. 
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
    backward_prob = np.empty((N, M))
    # INIT
    backward_prob[:, -1] = 0.0
    # BACKWARD PASS
    for n in range(M-2, -1, -1):
        for j in range(N):
            backward_prob[j, n] = logsumexp(transmat[j, :-1] + emlike[:, n+1] + backward_prob[:, n+1])
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
    for i in range(N):
        vi[i,0] = emlike[i,0] + startprob[i,0]
    print(emlike.shape)
#     vi[:,0] = emlike[:,0] + startprob[:,0]
    for t in range(1,M):
        for i in range(N):
            temp = vi[:,t-1] + transmat[:-1,i]
            vi[i,t] = np.max(temp) + emlike[i,t]
            path[i,t] = np.argmax(temp)
            # MAX PROB OF PREVIOUS VI-timestep * P(we go from each of prevois states to i) * P(we obeserve i at timestep t). (use + insted of *, since log domain)
#             vi[i,t] = np.max(vi[:,t-1] + transmat[:-1,i] , axis = 1) + emlike[i,t]
#             path[i,t] = np.argmax(vi[:,t-1] + transmat[:-1,i])
    zt = np.argmax(vi[:,-1])
    obsseq[M-1] = zt
    for t in range(M-1,0,-1):
        zt = path[int(zt),t]
        obsseq[t-1] = zt
    
#     vit[-1,argmax(1)[-1]], vit.argmax(1)
    
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
    # print(log_alpha.shape)
    log_gamma = np.empty(log_alpha.shape)
    M, N = log_alpha.shape
    for n in range(N):
        for i in range(M):
            log_gamma[i, n] = log_alpha[i, n] + log_beta[i, n] - logsumexp(log_alpha[:, N-1])
    return log_gamma

def updateMeanAndVar(X, gamma, varianceFloor=5.0):
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
#     X = 86,13
#     means, covars = 15,13
#     gamma = 15,86

    N, T = gamma.shape
    D = X.shape[1]
    
    numer = np.exp(gamma).dot(X)
    denom = np.exp(gamma).sum(1).reshape(-1,1)
    means = numer / denom
    
    covars = np.zeros((N, D))
    for state in range(N):
        for obs in range(D):
            numer = 0
            W =  (X[:, obs] - means[state, obs])**2
            numer = np.exp(gamma[state,:]).dot(W.reshape(-1,1))    
            covars[state,obs] = numer[0] / logsumexp(gamma[state,:])
            
    covars = np.where(covars < varianceFloor, varianceFloor, covars)
    return means, covars


def baum_welch(data,wordHMMs):
    orig_means, orig_covars = wordHMMs['means'], wordHMMs['covars']
    converging = True
    scores = []
    for _ in range(20):
        obsloglike = log_multivariate_normal_density_diag(data,wordHMMs['means'],wordHMMs['covars'])
        alpha = forward(obsloglike.T, np.log(wordHMMs['startprob']), np.log(wordHMMs['transmat'])) 
        beta = backward(obsloglike.T, np.log(wordHMMs['startprob']), np.log(wordHMMs['transmat']))
        scores.append(logsumexp(alpha.T[-1,:].T)) 
        print(logsumexp(alpha.T[-1,:].T))        
        gamma = statePosteriors(alpha, beta)
        means, covars = updateMeanAndVar(data, gamma)
        wordHMMs['means'] = means
        wordHMMs['covars'] = covars
        converging = False
    return scores

def plot_obsloglike(obsloglike):
    plt.figure()
    plt.pcolormesh(obsloglike.T)
#     plt.xlabel('features')
#     plt.ylabel('features')
    plt.colorbar()

def compute_scores(wordHMMs, data):
    scores = np.zeros((len(wordHMMs), len(data)))
    for i,model in enumerate(wordHMMs):
        for j,utterance in enumerate(data):
            obslike = log_multivariate_normal_density_diag(utterance['lmfcc'],
                                                        wordHMMs[model]['means'],
                                                        wordHMMs[model]['covars'])
            alphas = forward(obslike.T,np.log(wordHMMs[model]['startprob']),np.log(wordHMMs[model]['transmat']))
            scores[i,j] = logsumexp(alphas.T[-1,:].T)
    return alphas, scores

def compute_scores_viterbi(wordHMMs, data):
    scores = np.zeros((len(wordHMMs), len(data)))
    for i,model in enumerate(wordHMMs):
        for j,utterance in enumerate(data):
            obsloglike = log_multivariate_normal_density_diag(utterance['lmfcc'],
                                                        wordHMMs[model]['means'],
                                                        wordHMMs[model]['covars'])
#             alphas = forward(obslike.T,np.log(wordHMMs[model]['startprob']),np.log(wordHMMs[model]['transmat']))
            vi, _ = viterbi(obsloglike.T, np.log(wordHMMs[model]['startprob']), np.log(wordHMMs[model]['transmat'])) 
            scores[i,j] = np.max(vi.T[-1,:].T)
    return vi, scores


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
    obsloglike = log_multivariate_normal_density_diag(example['lmfcc'],
                                            wordHMMs['o']['means'],
                                            wordHMMs['o']['covars'])
    alpha_mat = forward(obsloglike.T, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat'])) 
    print(logsumexp(alpha_mat.T[-1,:]))
    print(example['loglik'])

    # SCORE ALL 44 UTTERANCES
    alphas, scores = compute_scores(wordHMMs, data)
    plt.figure()
    plt.title('alpha prediction')
    pred = np.argmax(scores,axis = 0)
    print('alpha scoring {}'.format(pred))
    plt.plot(pred)
    plt.show()
#     for d in scores.keys():
#         plt.scatter(d,d, c='b', label='true')
#         o = sorted(scores[d], key=lambda x: x[1],reverse=True)
#         plt.scatter(d, o[0][0], c='r', label='prediction')
#     plt.legend()
    plt.show()
# 

    # LOGLIKELIHOOD
    loglike = logsumexp(alpha_mat.T[-1,:].T)
    vi, obsseq = viterbi(obsloglike.T, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat'])) 
#     plt.pcolormesh(alpha_mat)
#     plt.plot(obsseq)
#     plt.show()
    print(example['vloglik'][0])
    print(np.max(vi.T[-1,:]))

    # SCORE ALL 44 UTTERANCES USING viterebi
    vi, scores = compute_scores_viterbi(wordHMMs, data)
    pred = np.argmax(scores, axis=0)
    print('viterbi scoring {}'.format(pred))
    plt.figure()
    plt.title('viterbi prediction')
    plt.plot(pred)
#     for d in scores.keys():
#         plt.scatter(d,d, c='b', label = 'true')
#         o = sorted(scores[d], key=lambda x: x[1],reverse=True)
#         plt.scatter(d, o[0][0], c='r', label='true')
#     plt.legend()
    plt.show()

#     beta_mat = backward(obsloglike.T, np.log(wordHMMs['4']['startprob']), np.log(wordHMMs['4']['transmat']))
#     
#     # CALC POSTERIORS
#     gamma_mat = statePosteriors(alpha_mat, beta_mat)
#     print(np.exp(gamma_mat).sum(0)) # summing over states. all=1. we will go to SOME state in each timestep.
#     print(np.exp(gamma_mat).sum(1)) # summing over timesteps. expected number of times we are in state i.
#     print(np.exp(gamma_mat).sum()) 
#     
#     # UPDATE MEAN AND VARIANCE - BAUM WELCH
#     print("SCORES: 4 0N 4")
#     scores4_4 = baum_welch(data[10]['lmfcc'], wordHMMs['4'])
#     print("\n SCORES: 4 0N 9")
#     scores4_9 = baum_welch(data[11]['lmfcc'], wordHMMs['9'])
#     plt.plot(scores4_4, label='4_on_4a')
#     plt.plot(scores4_9, label='4_on_4b')
#     plt.legend()
#     plt.show()
    return 0


if __name__=='__main__':
    main()

