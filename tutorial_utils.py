import numpy as np
import scipy.stats as stats
from scipy.linalg import eig
from functools import reduce

# Transition matrices for even process
even_Ts = {
    0: np.array([[0.5, 0],[0, 0]]),
    1: np.array([[0, 1.0],[0.5, 0]])
}

# Transition matrices for golden mean process
golden_Ts = {
    0: np.array([[0, 0],[0.5, 0]]),
    1: np.array([[0.5, 1.0],[0, 0]])
}

# Transition matrices for simple nonunifilar source
sns_Ts = {
    1: np.array([[0, 0.5],[0, 0]]),
    0: np.array([[0.5, 0],[0.5, 0.5]])
}

# Transition matrices for nemo process
nemo_Ts = {
    0: np.array([[0, 0, 0.5], [0.5, 0, 0], [0, 1, 0]]),
    1: np.array([[0.5, 0, 0.5], [0, 0, 0], [0, 0, 0]])
}

transition_matrices = dict(
    even=even_Ts,
    golden=golden_Ts,
    sns=sns_Ts,
    nemo=nemo_Ts
)

def gen_hmm(Ts,num=50,return_states=False):
    # Generate an HMM sample of length num, given the transition matrices as a dictionary.
    T = Ts[0]+Ts[1]
    eigs,vecs = eig(T,right=True,left=False)
    pi = np.real(vecs[:,np.argmax(np.abs(eigs))]/np.sum(vecs[:,np.argmax(np.abs(eigs))]))
    output = []
    v = pi
    if return_states:
        states = [v]
    for j in (range(num)):
        probs = np.array([(Ts[0]@v).sum(),(Ts[1]@v).sum()])
        x = np.random.choice(np.array([0,1]),p=probs)
        output+=[x]
        v = Ts[x]@v/(Ts[x]@v).sum()
        if return_states:
            states = [v]
    if return_states:
        return np.array(output), np.array(states)
    else: 
        return np.array(output)

def gen_anbn(n,mu=1):
    rv = stats.poisson(mu=mu)
    lens = rv.rvs(size=n)+1
    return np.array(reduce(lambda x,y:x+y,[[0]*k+[1]*k for k in lens],[]))