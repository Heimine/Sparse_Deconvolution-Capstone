import numpy as np
from scipy.ndimage import convolve

def Log_map(Z,D):
    proj_a = lambda w, z: z - np.inner(w, z) * w
    n, K = Z.shape
    T = np.zeros([n, K])
    for k in range(K):
        alpha = np.arccos(np.inner(Z[:, k], D[:, k]))
        proj_tem = proj_a(Z[:, k], D[:, k])
        T[:, k] = proj_temp * alpha / np.sin(alpha)
    
    return T

# -------------------------------
def Retract(Z, D, t):
    n, K = Z.shape
    T = np.zeros([n, K])

    for k in range(K):
        T[:, k] = Z[:, k] * np.cos(t[k]) + (D[:, k] / t[k]) * np.sin(t[k])
    T = T / np.linalg.norm(T, axis = 0) # Normalize T by column

    return T

# -------------------------------
def backtracking(y, A, X, fx, grad_fx, lamb, t, opts):
    # update X via backtracking linesearch
    m = np.max(y.shape)
    Q = lambda Z, tau: fx + np.linalg.norm(lamb * Z, 1) + innerprod(grad_fx, Z - X) \
        + 0.5 / tau * np.linalg.norm(Z - X, "fro") ** 2
    t = 8 * t
    X1 = soft_thres(X - t * grad_fx, lamb * t) # Proximal mapping
    if opts["isnonnegative"]:
        X1 = np.max(X1, 0)

    if opts["hard_thres"]:
        ind = (X1 <= opts["hard_threshold"])
        X1[ind] = 0

    while Psi_val(y, A, X1, lamb) > Q(X1,t):
        t = 1 / 2 * t
        X1 = soft_thres(X - t * grad_fx, lamb * t)
        if opts["isnonnegative"]:
            X1 = np.max(X1, 0)
        if opts["isupperbound"]:
            X1 = np.min(X1, opts["upperbound"])
        if opts["hard_thres"]:
            ind = (X1 <= opts["hard_threshold"])
            X1[ind] = 0
    
    return [X1, t]

def innerprod(U, V):
    # This function is a dependency for the backtracking function
    return np.sum(U * V)

def Psi_val(y, A, Z, lamb = None):
    # This function is a dependency for the backtracking function
    # Also a dependency for the linesearch function
    m = np.max(y.shape)
    n, K = A.shape
    y_hat = np.zeros(y.shape)

    for k in range(K):
        ### Involving circular convolution, need to ask
        y_hat = y_hat + cconv(A[:, k], Z[:, k], m)

    if lamb is None:
        f = 0.5 * np.sum((y - y_hat) ** 2)
    else:
        f = 0.5 * np.linalg.norm(y - y_hat) ** 2 + np.linalg.norm(lamb * Z, 1)

    return f

# -------------------------------
def compute_error(A, X, opts):
    n, K = A.shape
    m, m_hat = X.shape

    A_0 = np.vstack([np.zeros([int(n / 3), K]), opts["A_0"],
                     np.zeros([int(n / 3), K])]) # Why
    X_0 = opts["X_0"]
    err_A = 0
    err_X = 0
    for i in range(K):
        a = A[:, i]
        x = X[:, i]
        cor = np.zeros([K, 1])
        ind = np.zeros([K, 1])
        for j in range(K):
            #### Circular convolution again
            Corr = cconv(reversal(A_0[:, j]), a, m)
            cor[j] = np.max(np.abs(Corr), axis = 0)
            ind[j] = np.argmax(np.abs(Corr), axis = 0)
        Ind = np.argmax(cor)
        # Use np.roll to mimic circshift
        a_max = np.roll(A_0[:, Ind], int(ind[Ind] - 1), axis = 0)
        x_max = np.roll(X_0[:, Ind], int(-(ind[Ind] - 1)), axis = 0)
        err_A = err_A + np.min([np.linalg.norm(a_max - a), np.linalg.norm(a_max + a)])
        err_X = err_X + np.min([np.linalg.norm(x_max - x), np.linalg.norm(x_max + x)])
        
    return [err_A, err_X]

# -------------------------------
def compute_gradient(A, X, y_b, y_hat, gradient_case):
    # Compute (Riemannian) gradient
    proj_a = lambda w, z: z - np.inner(w, z) * w
    m, K = X.shape
    n, n_hat = A.shape

    if gradient_case == 0:
        Grad = np.zeros([m, K])
    elif gradient_case == 1:
        Grad = np.zeros([n, K])

    for k in range(K):
        if gradient_case == 0:
            #### Circular convolution alert
            Grad[:, k] = cconv(reversal(A[:, k], m), y_hat - y_b, m).flatten()
        elif gradient_case == 1:
            G = cconv(reversal(X[:, k], m), y_hat - y_b, m).flatten()
            Grad[:, k] = proj_a(A[:, k], G[:n])
    
    return Grad

# -------------------------------
def compute_y(A, X):
    # compute y = sum_k conv(a_k,x_k)
    m, K = X.shape
    y_hat = np.zeros([m, 1])
    for k in range(K):
        ### Circular convolution alert
        y_hat = y_hat + cconv(A[:, k], X[:, k], m)
    
    return y_hat

# -------------------------------
def gen_data(theta, m, n, b, noise_level, a_type,x_type):
    # generate the groudtruth data
    # y = sum_{k=1}^K a0k conv x0k + b*1 + n
    # s = rng(seed)
    
    # generate the kernel a_0
    gamma = [1.7, -0.712] # Parameter for AR2 model
    t = np.linspace(0, 1, n).reshape([n, 1]) # [0:1/(n-1):1]'
    case = a_type.lower()
    if case == "randn": # Random Gaussian
        a_0 = np.random.normal(size = [n, 1])
    elif case == "ar2": # AR2 kernel
        tau = 0.01 * ar2exp(gamma) # Function defined below
        a_0 = np.exp(-t / tau[0]) - np.exp(-t/tau[1])
    elif case == "ar1": # AR1 model
        tau = 0.25
        a_0 = np.exp(-t / tau)
    elif case == "gaussian":
        t = np.linspace(-2, 2, n).reshape([n, 1])
        a_0 = np.exp(-t**2)
    elif case == "sinc":
        sigma = 0.05
        a_0 = np.sinc((t-0.5)/sigma)
    else:
        raise ValueError("Wrong type")

    a_0 = a_0 / np.linalg.norm(a_0, axis = 0)  # Normalize kernel by column

    # Generate the spike train x_0
    case_x = x_type.lower()
    if case_x == "bernoulli":
        x_0 = (np.random.uniform(size = [m, 1]) <= theta) # Bernoulli spike train
    elif case_x == 'bernoulli-rademacher':
        x_0 = (np.random.uniform(size = [m, 1]) <= theta) * ((np.random.uniform(
            size = [m, 1]) < 0.5) - 0.5) * 2
    elif case_X == 'bernoulli-gaussian':
        # Gaussian-Bernoulli spike train
        x_0 = np.random.normal([m, 1]) * (np.random.uniform(m, 1) <= theta)
    else:
        raise ValueError("Wrong type")

    # generate the data y = a_0 conv b_0 + bias + noise
    ##### Circular convolution alert
    y_0 = cconv(a_0, x_0, m) + b * np.ones([m,1])
    y = y_0 + np.random.normal(size = [m, 1]) * noise_level
        
    return [a_0, x_0, y_0, y]

# -------------------------------
def ar2exp(g):
    # get parameters of the convolution kernel for AR2 process
    # Dependency of gen_data
    if len(g) == 1:
        g.append(0)
    temp = np.roots([1, -g[0], -g[1]]) # Polynomial roots
    d = np.max(temp)
    r = np.min(temp)
    tau_d = -1 / np.log(d)
    tau_r = -1 / np.log(r)

    tau_dr = [tau_d, tau_r]
    return tau_dr

# -------------------------------
def linesearch(y, A, X, fa, grad_a):
    # update A via Riemannian linsearch
    m = np.max(y.shape)
    K_hat, K = A.shape
    eta = 0.8
    tau = 1

    norm_grad = np.linalg.norm(grad_a, "fro")
    Norm_G = np.zeros([K, 1])
    for k in range(K):
        Norm_G[k] = np.linalg.norm(grad_a[:, k])

    A1 = Retract(A, -tau * grad_a, tau * Norm_G)

    count = 1
    while Psi_val(y, A1, X) > fa - eta * tau * norm_grad ** 2:
        tau = 0.5 * tau
        A1 = Retract(A, -tau*grad_a, tau*Norm_G)

        if count >= 100:
            break
        count += 1
    
    return [A1, tau]

# -------------------------------
def reversal(X, m = None):
    if len(X.shape) == 1:
        X = X.reshape([X.shape[0], 1])
    if m != None:
        X = np.vstack([X[:np.min([X.shape[0], m]), :], np.zeros(
            [np.max([m - X.shape[0], 0]), X.shape[1]])])
        
    revX = np.vstack([X[0, :], np.flipud(X[1:, :])])
    
    return [revX]

# -------------------------------
def shift_correlation(a, x, opts):
    a_0 = opts["A_0"]
    x_0 = opts["x_0"]

    n_0 = np.max(a_0.shape)
    n = np.max(a.shape)
    m = np.max(x.shape)

    if opts["ground_truth"]:
        ###### Circular convolution alert
        Corr = cconv(reversal(a_0), a, m)
        ind_hat, ind = np.max(np.abs(Corr))
        Corr_max = Corr[ind]

        if Corr_max > 0:
            ## not for sure since circshift shift the first dim that's not 1
            a_shift = np.roll(a, ind - 1)
            x_shift = np.roll(x, -(ind - 1))
        else:
            a_shift = -np.roll(a, ind - 1)
            x_shift = np.roll(x, -(ind - 1))
            # a_max = np.roll(A_0[:, Ind], ind[Ind] - 1, axis = 0)
    
    return [a_shift, x_shift]

# -------------------------------
def soft_thres(z, lamb):
    z = np.sign(z) * np.max(np.abs(z) - lamb, 0)
    
    return z

# -------------------------------
def cconv(vec1, vec2, length):
    # Since there's a lot of functions use circular function
    # and python doesn't have a function for that
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return (np.fft.ifft(np.fft.fft(vec1.flatten(), length)
                        \* np.fft.fft(vec2.flatten(), length))).reshape([length,1])
   




