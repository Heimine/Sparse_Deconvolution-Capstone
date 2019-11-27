import nets_sort
import loss_fns
import torch
from torch.autograd import Variable
import torch.optim as optim
import empirical_sim
import prior
import torch.nn as nn
import simple_noise
import generative_model
import time
import sys
import gen_data_2D

# Parameters
batch_size = 256//2#256*10
eval_batch_size = 256
iters_per_eval = 10#10#*4*4
stepsize = 1E-3
_2D = True

if _2D:
    print("Using 2D loss.")

kernel_sigmas = [64.0, 320.0, 640.0, 1920.0]
# kernel_sigmas = [1.0]

use_cuda = False
warmstart = False

# # Construct a generative model
# p = prior.UniformCardinalityPrior([0,0,-750], [6400,6400,750], 1000.0, 7000.0, 5)
# sim = empirical_sim.EmpiricalSim(64,6400,empirical_sim.load_AS())
# noise = simple_noise.EMCCD(100.0)
# gen_model = generative_model.GenerativeModel(p,sim,noise)

# #Note that queues do not work on OS X.
# if sys.platform != 'darwin':
#     m = generative_model.MultiprocessGenerativeModel(gen_model,4,batch_size)
# else:
#     m = gen_model

# Construct the network
# Warmstart?
if warmstart:
    net = torch.load("net_AS")
else:
    net = nets_sort.DeepLoco() if not _2D else nets_sort.DeepLoco(min_coords = [0,0], max_coords = [64,64])

if use_cuda:
    net = net.cuda()

theta_mul = Variable(torch.Tensor([1.0,1.0]))
if use_cuda:
    theta_mul = theta_mul.cuda()

# Takes a CPU batch and converts to CUDA variables
# Also zero/ones the simulated weights
def to_v(d):
    v = Variable(torch.Tensor(d))
    if use_cuda:
        v = v.cuda()
    return v

def to_variable(theta, weights, images, masks):
    return to_v(theta) ,to_v(weights).sign_() ,to_v(images), to_v(masks)


########################
print("Random generated image")
arg = {"n": 2,
       "b": 0.3,
       "T": 10,
       "noise_level": 0.08,
       "a_type": "randn",
       "raw_data_handeling": "max_0"}
arg["theta"] = 0.05
arg["xgrid"] = [64, 64]
arg["x_type"] = 'bernoulli-gaussian'
    
# images, thetas, weights = sample(2, 64, 5, 2)

########################


eval_batch_size = 2

# Generate an evaluation batch
(e_images, e_theta, e_weights, e_masks) = gen_data_2D.sample(arg, eval_batch_size, 64, 5, 2)
print(e_theta)
(e_theta, e_weights, e_images, e_masks) = to_variable(e_theta, e_weights, e_images, e_masks)
# print(e_images.shape)
# print(e_theta.shape)
# print(e_weights.shape)
# print("input image:")
# print(e_images)
# print(e_theta)
# print(e_weights)
# print("...........")

# import torch.nn as nn


def loss_fn(o_theta, o_w, theta, weights):
    if _2D:
        theta = theta[:,:,:2]
    return loss_fns.multiscale_l1_laplacian_loss(o_theta, o_w,
                                                 theta, weights,
                                                 kernel_sigmas).mean()

# lr_schedule = [(stepsize, 200), (stepsize/100, 100)]
# lr_schedule = [(stepsize, 5), (stepsize/5, 5), (stepsize/25, 5), (stepsize/100, 5)]
lr_schedule = [(stepsize, 5)]




for stepsize, iters in lr_schedule:
    # Constuct the optimizer
    print("stepsize = ", stepsize)
    optimizer = optim.Adam(net.parameters(),lr=stepsize)
    for i in range(iters):
        iter_start_time = time.time()
        print("iter",i)
        # Compute eval
        net.eval()
        (o_theta_e, o_w_e) = net(e_images)

        e_loss = loss_fn(o_theta_e, o_w_e, e_theta,e_weights)


        # print("\teval", e_loss.data.item())
        print("\teval", e_loss)
        # print(o_theta_e)
        s_time = time.time()
        for batch_idx in range(iters_per_eval):
            #print(".")
            net.train()

            # (images, theta, weights, masks) = gen_data_2D.sample(arg, batch_size, 64, 5, 2)
            (theta, weights, images, masks) =  (e_theta, e_weights, e_images, e_masks)


            theta, weights, images, masks = to_variable(theta, weights, images, masks)
            
            print(images)
            # print(theta)
            # print(weights)


            (o_theta, o_w) = net(images)

            

            train_loss = loss_fn(o_theta, o_w, theta, weights)


            print("\ttrain", train_loss)

            train_loss.backward()
            optimizer.step()
        torch.save(net.cpu(), "net_sim_sorted")
        if use_cuda:
            net.cuda()
        # print("A:", time.time()-iter_start_time)

print(theta)
print(weights)
net.eval()
(o_theta, o_w) = net(images)

print(o_theta)
print(o_w)
