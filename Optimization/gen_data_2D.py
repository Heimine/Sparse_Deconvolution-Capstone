# generate the groudtruth data
# y = sum_{k=1}^K a0k conv x0k + b*1 + n
# s = rng(seed)
def gen_data_2D(theta, x_grid, n, b, noise_level, a_type,x_type, raw_data_handeling = None):
    """
    Parameters:
    ----------------
    theta: the threshold for generating spike train X_0
    x_grid: a length 2 list denoting the grid dimension for X
    n: the dimension of the kernel (we use squared kernel)
    b: the magnitude of the bias term
    noise_level: the magnitude of noise
    a_type: the kernel type we want to apply on X (randn, 2d-gaussian, sinc)
    x_type: which method we want to generate x_0 (bernoulli, bernoulli-gaussian)
    raw_data_handeling: our generated X may have some entry as negative values,
                        however, this will not happen in real image, so we need
                        to take care of those negative entries. (max_0, sigmoid)
    """
    
    # generate the kernel a_0
    gamma = [1.7, -0.712] # Parameter for AR2 model
    t = np.linspace(0, 1, n**2).reshape([n, n]) 
    case = a_type.lower()
    if case == "randn": # Random Gaussian
        a_0 = np.random.normal(size = [n, n])
    elif case == "sinc":
        sigma = 0.05
        a_0 = np.sinc((t-0.5)/sigma)
    elif case == "2d-gaussian":
        sigma = 0.5 # could perturb sigma if you want
        # 2D gaussian kernel
        grid_hori = np.linspace(-n / 2, n / 2, n)
        grid_verti = grid_hori.copy()
        mesh_x, mesh_y = np.meshgrid(grid_hori, grid_verti)
        a_0 = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(mesh_x ** 2 + mesh_y ** 2) / 2 * sigma ** 2)
    else:
        raise ValueError("Wrong type")

    #a_0 = a_0 / np.linalg.norm(a_0, axis = 0)  # Normalize kernel by column
    a_0 = a_0 / np.max(np.linalg.eig(a_0)[0])

    # Generate the spike train x_0
    m_0, m_1 = x_grid
    case_x = x_type.lower()
    if case_x == "bernoulli":
        x_0 = (np.random.uniform(size = [m_0, m_1]) <= theta).astype(int) # Bernoulli spike train
    elif case_x == 'bernoulli-gaussian':
        # Gaussian-Bernoulli spike train
        x_0 = np.random.normal(size = [m_0, m_1]) * (np.random.uniform(size = [m_0, m_1]) <= theta)
    else:
        raise ValueError("Wrong type")
    # Now handle negative entries
    if case_x != "bernoulli":
        case_handle = raw_data_handeling.lower()
        if case_handle == "max_0":
            x_0 = np.maximum(x_0, 0)
        elif case_handle == "sigmoid":
            indices = (x_0 < 0)
            x_0[indices] = 1 / (1 + np.exp(-x_0[indices])) 

    # generate the data y = a_0 conv b_0 + bias + noise
    ##### Circular convolution alert
    y_0 = cconv(a_0, x_0, [m_0, m_1]) + b * np.ones([m_0,m_1])
    y = y_0 + np.random.normal(size = [m_0, m_1]) * noise_level
        
    return [a_0, x_0, y_0, y]

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
    return np.array(tau_dr)

def cconv(mat1, mat2, output_shape):
    # Since there's a lot of functions use circular function
    # and python doesn't have a function for that
    # Dependency of gen_data
    
    return np.real((np.fft.ifft2(np.fft.fft2(mat1, s = output_shape) \
                        * np.fft.fft2(mat2, s = output_shape))))
