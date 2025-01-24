import numpy as np
import torch
import torch.nn.functional as F
import os

def search_hypers(sigmas):
    sigmas = sigmas[:-1]
    sigmas_max = max(sigmas)


    v2_list = np.arange(50, 1001, 50)
    v3_list = np.arange(10, 101, 10)
    v1_list = np.linspace(0.001, 0.009, 9)
    zero_count_default = 0
    index_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

    for v1 in v1_list:
        for v2 in v2_list:
            for v3 in v3_list:
                flag = True
                lambda_t_list = []
                for sigma in sigmas:
                    sigma_n = sigma/sigmas_max
                    temp_cond_indices = [0]
                    for tau in range(25):
                        if tau not in index_list:
                            lambda_t_list.append(1)
                        else:
                            tau_p = 0 
                            tau_ = tau/24

                            Q = v3 * abs((tau_-tau_p)) - v2*sigma_n
                            k = 0.8
                            b = -0.2

                            lambda_t_1 = (-(2*v1 + k*Q) + ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                            lambda_t_2 = (-(2*v1 + k*Q) - ((2*k*v1+k*Q)**2 - 4*k*v1*(k*v1+Q*b))**0.5)/(2*k*v1)
                            v1_ = -v1
                            lambda_t_3 = (-(2*v1_ + k*Q) + ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                            lambda_t_4 = (-(2*v1_ + k*Q) - ((2*k*v1_+k*Q)**2 - 4*k*v1_*(k*v1_+Q*b))**0.5)/(2*k*v1_)
                            try:
                                if np.isreal(lambda_t_1):
                                    if lambda_t_1 >1.0:
                                        lambda_t = lambda_t_1
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                if np.isreal(lambda_t_2):
                                    if lambda_t_2 >1.0:
                                        lambda_t = lambda_t_2
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                if np.isreal(lambda_t_3):
                                    if lambda_t_3 <=1.0 and lambda_t_3>0:
                                        lambda_t = lambda_t_3
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                if np.isreal(lambda_t_4):
                                    if lambda_t_4 <=1.0 and lambda_t_4>0:
                                        lambda_t = lambda_t_4
                                        lambda_t_list.append(lambda_t/(1+lambda_t))
                                        continue
                                flag = False
                                break
                            except:
                                flag = False
                                break
                            lambda_t_list.append(lambda_t/(1+lambda_t))
       
                if flag == True:
                    zero_count = sum(1 for x in lambda_t_list if x > 0.5)
                    if zero_count > zero_count_default:
                        zero_count_default = zero_count
                        v_optimized = [v1,v2,v3]
                        lambda_t_list_optimized = lambda_t_list
                    
    X = np.array(sigmas)

    Y = np.arange(0,25,1)
    temp_i = np.array(temp_cond_indices)
    X, Y = np.meshgrid(X, Y)
    lambda_t_list_optimized = np.array(lambda_t_list_optimized)
    lambda_t_list_optimized = lambda_t_list_optimized.reshape([len(sigmas),25])

    lambda_t_list_optimized = torch.tensor(lambda_t_list_optimized)
    Z= lambda_t_list_optimized

    z_upsampled = F.interpolate(Z.unsqueeze(0).unsqueeze(0), scale_factor=10, mode='bilinear', align_corners=True)

    return lambda_t_list_optimized
