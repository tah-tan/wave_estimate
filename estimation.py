# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np

# 元データ生成
x = np.arange(-3, 3, .1)
t = []
for xt in x:
    tt = np.sin(xt) + 0.1 * np.random.randn()
    t.append(tt)

# 簡単のため、モデルは3次の多項式とする
# y = w0 + w1 * x + w2 * x^2 + w3 * x^3
#   phi0 = 1
#   phi1 = x
#   phi2 = x^2
#   phi3 = x^3
# パラメータベクトル（配列） w = [w0, w1, w2, w3]
# パラメータベクトルの推定値 m
# 横軸 x
# 観測値 y
# 正規方程式 Phi[i][j] = phij(x_i)

# 初期値
alpha = 1.0
beta = 1.0
M = 4           # 次元

I = np.matrix(np.identity(M))   # 単位行列

# 正規方程式の初期化
Phi = np.zeros([len(x), M])
for i in range(0, len(x)):
    for j in range(0, M):
        Phi[i][j] = x[i] ** j

Phi_T_Phi = np.dot(Phi.T, Phi)

## パラメータの推定
alpha_pre = 0.0
beta_pre = 0.0
epsilon = 1.0e-10
m = np.zeros(M)
for loop_num in range(0, 10):
    print("--- loop num (" + str(loop_num) + ") ---")
    # Sの推定
    S_inv = np.dot(alpha, I) + np.dot(beta, Phi_T_Phi)
    S = np.linalg.inv(S_inv)
    
    # mの推定
    m_tmp1 = np.dot(Phi.T, t)
    m_tmp2 = np.dot(S, m_tmp1)
    m_tmp = np.dot(beta, m_tmp2)
    # 何故かこれをそのままmにすると内積がエラーになるので、いったん別途配列を作って渡す
    for i in range(0, M):
        m[i] = m_tmp[0,i]
    print("m = " + str(m))
    
    ## 正規化係数α、βの再計算
    # 固有値分解
    beta_Phi_T_Phi = np.dot(beta, Phi_T_Phi)
    lambda_, u_ = np.linalg.eig(beta_Phi_T_Phi)
    
    # αの再計算
    gamma = 0
    for lambda_tmp in lambda_:
        gamma = gamma + (lambda_tmp / (lambda_tmp + alpha))
    alpha = gamma / np.dot(m, m)
    print("alpha = " + str(alpha))
    
    # βの再計算
    beta_tmp = 0
    for i in range(0, len(x)):
        m_phi = 0
        for j in range(0, M):
            m_phi = m_phi + m[j] * (x[i] ** j)
        beta_tmp = beta_tmp + (t[i] - m_phi) ** 2
    beta_inv = beta_tmp / (len(x) - gamma)
    beta = 1 / beta_inv
    print("beta = " + str(beta))

    # α、βの差分を計算
    alpha_dif = np.abs(alpha - alpha_pre)
    beta_dif = np.abs(beta - beta_pre)
    print("alpha_dif = " + str(alpha_dif))
    print("beta_dif = " + str(beta_dif))
    alpha_pre = alpha
    beta_pre = beta

    if (alpha_dif < epsilon) and (beta_dif < epsilon):
        print("ループ終了条件に合致")
        break

# 近似曲線の表示
y = []
for xt in x:
    yt = 0
    for i in range(0, M):
        yt = yt + m[i] * (xt ** i)
    y.append(yt)


# 結果表示
plt.plot(x, t)
plt.plot(x, y)
plt.show()

