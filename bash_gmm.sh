#!/bin/bash

# configuration for a mixture of Gaussians

epoch=1000
dataset=gmm
train_size=10000
batch_size=10000

cg_maxiter=20

save_iter=2
print_iter=2

# # gradient descent ascent
# d_optim=gd
# d_step_size=0.01
# d_num_step=1
# g_optim=gd
# g_step_size=0.01
# simultaneous=1

# # total gradient descent ascent, a.k.a. stackelberg dynamics in Fiez et al 2019
# d_optim=gd
# d_step_size=0.02
# d_num_step=1
# g_optim=sd
# g_step_size=0.01
# simultaneous=1

# # follow-the-ridge
# d_optim=fr
# d_step_size=0.02
# d_num_step=1
# g_optim=gd
# g_step_size=0.01
# simultaneous=1

# # gd-newton
# d_optim=newton
# d_step_size=1.0
# d_num_step=1
# g_optim=gd
# g_step_size=0.01
# simultaneous=0

# complete newton
d_optim=newton
d_step_size=1.0
d_num_step=1
g_optim=newton
g_step_size=0.1
simultaneous=0
cg_maxiter_cn=32
cg_lam_cn=0.1


python run.py --epoch $epoch \
              --dataset $dataset \
              --train_size $train_size \
              --batch_size $batch_size \
              --pretrain 1 \
              --d_optim $d_optim \
              --d_step_size $d_step_size \
              --d_num_step $d_num_step \
              --g_optim $g_optim \
              --g_step_size $g_step_size \
              --cg_maxiter $cg_maxiter \
              --cg_maxiter_cn $cg_maxiter_cn\
              --cg_tol 1e-40 \
              --cg_lam 0.0 \
              --cg_lam_cn $cg_lam_cn\
              --simultaneous $simultaneous \
              --save_iter $save_iter \
              --print_iter $print_iter
