
from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        for s in range(S):
            alpha[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            Xt = self.obs_dict[Osequence[t]]
            for s in range(S):
                sum_a_alpha = 0
                for s_dash in range(S):
                    sum_a_alpha = sum_a_alpha + self.A[s_dash][s] * alpha[s_dash][t - 1]
                alpha[s][t] = self.B[s][Xt] * sum_a_alpha
        return alpha




    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        for s in range(S):
            beta[s][ L -1] = 1
        for t in reversed(range( L -1)):
            for s in range(S):
                for s_dash in range(S):
                    beta[s_dash, t] = sum \
                        ([beta[j, t + 1] * self.A[s_dash, j] * self.B[j, self.obs_dict[Osequence[t + 1]]] for j in range(S)])
        return beta


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        T = len(Osequence) - 1
        alpha = self.forward(Osequence)
        P = 0
        for s in range(len(self.pi)):
            P += alpha[s][T]
        return P


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        S = len(self.pi)
        L = len(Osequence)
        P = np.zeros([S, L])

        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        propTerm = sum(alpha[:, -1])

        for t in range(L):
            for s_dash in range(S):
                P[s_dash , t] = alpha[s_dash ,t] * beta[s_dash ,t ] /propTerm

        return P



    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] =
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        propTerm = sum(alpha[:, -1])

        for t in range( L -1):
            for s in range(S):
                for s_dash in range(S):
                    prob[s, s_dash, t] = self.A[s, s_dash] * self.B[s_dash, self.obs_dict[Osequence[t + 1]]] * beta[s_dash, t + 1] * alpha[s, t] / propTerm
        return prob


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.pi)
        L = len(Osequence)
        small_delta = np.zeros([S, L])
        big_delta = np.zeros([S, L], dtype="int")
        small_delta[: ,0] = self.pi * self.B[: ,self.obs_dict[Osequence[0]]]
        for t in range(1 ,L):
            for s_dash in range(S):
                small_delta[s_dash, t] = self.B[s_dash, self.obs_dict[Osequence[t]]] * np.max \
                    (self.A[:, s_dash] * small_delta[:, t- 1])
                big_delta[s_dash, t] = np.argmax(self.A[:, s_dash] * small_delta[:, t - 1])
        Z = np.argmax(small_delta[:, L - 1])
        path.append(Z)
        for t in range(L - 1, 0, -1):
            Z = big_delta[Z, t]
            path.append(Z)
        path = path[::-1]

        states = [0] * len(path)

        for i in self.state_dict:
            for j in range(len(path)):
                if path[j] == self.state_dict[i]:
                    states[j] = i

        return states

    # DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O

