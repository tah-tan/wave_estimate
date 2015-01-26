# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np

# naming rule : by PRML
# parameter vector: w = [w0, w1, w2, w3]
# order : M-1
# estimated value of parameter vector: m
# horizontal axis: x
# observed value: t 
# normal equation: Phi[i][j] = phij(x_i)

## parameters and customisable functions (need customise)
def get_horizontal_val():
    return np.arange(-3, 3, .1)


def get_observed_val(x_):
    t_ = []
    for xt in x_:
        tt = np.sin(xt) + 0.1 * np.random.randn()
        t_.append(tt)
    return t_


def get_basis_function(val, order):
    # porinominal function
    return (val ** order)


def get_estimated_val(x_, m_, M_):
    y_ = []
    for xt in x_:
        yt = 0
        for i in range(0, M_):
            yt = yt + m_[i] * get_basis_function(xt, i)
        y_.append(yt)
    return y_


def get_initial_alpha_beta():
    return 1.0, 1.0

 
## bayes's linear regression functions (do not need customise)
def get_m(Phi_, t_, S_, beta_, M_):
    m_tmp1 = np.dot(Phi_.T, t_)
    m_tmp2 = np.dot(S_, m_tmp1)
    m_tmp = np.dot(beta_, m_tmp2)
    # When this m_tmp is set to m and calculate inner product, error occurs.
    # So, make another array and set each member on it.
    print(m_tmp)
    m_ = np.zeros(M_)
    for i in range(0, M_):
        m_[i] = m_tmp[0,i].real
        # I don't know this should be real part or the norm of the number.
    return m_


def get_S(alpha_, beta_, Phi_T_Phi_, M_):
    I_ = np.matrix(np.identity(M_))     # Identity matrix
    S_inv = np.dot(alpha_, I_) + np.dot(beta_, Phi_T_Phi_)
    S_ = np.linalg.inv(S_inv)
    return S_


def get_gamma(beta_, Phi_T_Phi_, alpha_):
    beta_Phi_T_Phi = np.dot(beta_, Phi_T_Phi_)
    lambda_, u_ = np.linalg.eig(beta_Phi_T_Phi)

    gamma_ = 0
    for lambda_tmp in lambda_:
        gamma_ = gamma_ + (lambda_tmp / (lambda_tmp + alpha_))
    return gamma_


def get_alpha(gamma_, m_):
   return gamma_ / np.dot(m_, m_)


def get_beta(x_, t_, m_, M_, gamma_):
    beta_tmp = 0
    for i in range(0, len(x_)):
        m_phi = 0
        for j in range(0, M_):
            m_phi = m_phi + m_[j] * get_basis_function(x_[i], j)
        beta_tmp = beta_tmp + (t_[i] - m_phi) ** 2
    beta_inv = beta_tmp / (len(x_) - gamma_)
    return (1 / beta_inv)


def get_normal_equation(x_, M_):
    Phi_ = np.zeros([len(x_), M_])
    for i in range(0, len(x_)):
        for j in range(0, M_):
            Phi_[i][j] = get_basis_function(x_[i], j)
    return Phi_


epsilon = 1.0e-10
def estimate_param(x_, t_, M_):
    alpha, beta = get_initial_alpha_beta()
    Phi = get_normal_equation(x_, M_)
    Phi_T_Phi = np.dot(Phi.T, Phi)
    
    alpha_pre = 0.0
    beta_pre = 0.0
    for loop_count in range(0, 10):
        S = get_S(alpha, beta, Phi_T_Phi, M_)
        m_ = get_m(Phi, t_, S, beta, M_)
        gamma = get_gamma(beta, Phi_T_Phi, alpha)
        alpha = get_alpha(gamma, m_)
        beta = get_beta(x_, t_, m_, M_, gamma)
    
        alpha_dif = np.abs(alpha - alpha_pre)
        beta_dif = np.abs(beta - beta_pre)
        alpha_pre = alpha
        beta_pre = beta
    
        if (alpha_dif < epsilon) and (beta_dif < epsilon):
            S = get_S(alpha, beta, Phi_T_Phi, M_)
            m_ = get_m(Phi, t_, S, beta, M_)
            break

    log_evi1 = M_ * np.log(alpha) / 2
    log_evi2 = len(x_) * np.log(beta) / 2
    E1_tmp1_1 = t_ - np.dot(Phi, m_)
    E1_tmp1_2 = np.linalg.norm(E1_tmp1_1)
    E1_tmp1_3 = E1_tmp1_2 ** 2
    E1_tmp1 = beta * E1_tmp1_3 / 2
    E1_tmp2_1 = np.dot(m_, m_)
    E1_tmp2 = alpha * E1_tmp2_1 / 2
    E1 = E1_tmp1 + E1_tmp2
    log_evi3 = E1
    S_inv = np.linalg.inv(S)
    log_evi4 = np.log( np.linalg.det(S_inv) ) / 2
    log_evi5 = len(x_) * np.log(2 * np.pi) / 2
    log_evi_ = log_evi1 + log_evi2 - log_evi3 - log_evi4 - log_evi5

    return m_, log_evi_
 

## for program execution (your environment dependent)
def display_result(x_, t_, y_):
    plt.plot(x_, t_)
    plt.plot(x_, y_)
    plt.show()


## main function
def main():
    x = get_horizontal_val()
    t = get_observed_val(x)
    
    log_evi = -1000.0
    m = []
    M = 0
    for M_ in range(2, 20):
        m_, log_evi_ = estimate_param(x, t, M_)
        if (log_evi_ > log_evi):
            M = M_
            log_evi = log_evi_
            m = m_

    print(M, log_evi, m)
    y = get_estimated_val(x, m, M)
    display_result(x, t, y)


## this program starts from here
main()

