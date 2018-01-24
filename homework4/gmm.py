import json
import random
import numpy as np


def calc_gamma(x_i, pi, mu, cov, K):
    temp = []
    for k in range(0,K):
        x_muk = np.subtract(x_i, mu[k])
        sigma_k = np.reshape(cov[k],(2,2))
        num = pi[k] * np.exp((-0.5) *  x_muk.T.dot(np.linalg.inv(sigma_k)).dot(x_muk))
        deno = 2 * np.pi * np.sqrt(np.linalg.det(sigma_k))
        temp.append(num/deno)
    return np.array(temp)

def calc_mu(gamma_i, mu_denominator, x_i, K):
    temp = []
    for k in range(0,K):
        temp.append(np.multiply((gamma_i[k] / mu_denominator[k]), x_i))
    return temp

def calc_cov(x_i, mu, cov_deno, gamma_i, K):
    temp = []
    for k in range(0,K):
        diff = np.reshape(np.subtract(x_i,mu[k]), (2,1))
        temp.append(np.multiply ((gamma_i[k] / cov_deno[k]) , diff.dot(diff.T).reshape(4)))
    return temp;

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###

    # Run 100 iterations of EM updates
    for t in range(100):
        gamma = []
        mu_num = []
        cov_num = []
        
        for ind in range(0,len(X)):
            temp = calc_gamma(X[ind], pi, mu, cov, K);
            gamma.append(temp / np.sum(temp))
        gamma = np.array(gamma)
        pi = np.sum(gamma, axis = 0) / np.sum(gamma)

        mu_denominator = np.sum(gamma, axis = 0)
        for ind in range(0,len(X)):
            mu_num.append(calc_mu(gamma[ind], mu_denominator, X[ind], K))
        mu = np.sum(mu_num, axis = 0)

        cov_denominator = np.sum(gamma, axis = 0)
        for ind in range(0,len(X)):
            cov_num.append(calc_cov(X[ind], mu, cov_denominator, gamma[ind], K))
        cov = np.sum(cov_num, axis = 0)

    return mu.tolist(), cov.tolist()


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()