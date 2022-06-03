#!/bin/bash

# configuration for estimation the mean vector of a gaussian distribution

epoch=1000
# dataset=single_gaussian
dataset=single_gaussian_ill_conditioned
train_size=10000
batch_size=10000

cg_maxiter=8

save_iter=1
print_iter=1

# # 2ts-gda
# d_optim=gd
# d_step_size=0.5
# d_num_step=1
# g_optim=gd
# g_step_size=0.05
# simultaneous=0

# # gda-20
# d_optim=gd
# d_step_size=0.05
# d_num_step=20
# g_optim=gd
# g_step_size=0.05
# simultaneous=0

# # sd 
# d_optim=gd
# d_step_size=0.5
# d_num_step=1
# g_optim=sd
# g_step_size=0.05
# simultaneous=1

# # fr 
# d_optim=fr
# d_step_size=0.5
# d_num_step=1
# g_optim=gd
# g_step_size=0.05
# simultaneous=1

# # gd-newton
# d_optim=newton
# d_step_size=1.
# d_num_step=1
# g_optim=gd
# g_step_size=0.05
# simultaneous=0

# complete newton
d_optim=newton
d_step_size=1.
d_num_step=1
g_optim=newton
g_step_size=1.
simultaneous=0

python run.py --epoch $epoch \
              --dataset $dataset \
              --train_size $train_size \
              --batch_size $batch_size \
              --pretrain 0 \
              --d_optim $d_optim \
              --d_step_size $d_step_size \
              --d_num_step $d_num_step \
              --g_optim $g_optim \
              --g_step_size $g_step_size \
              --cg_maxiter $cg_maxiter \
              --cg_tol 1e-40 \
              --cg_lam 0. \
              --cg_lam_cn 0. \
              --simultaneous $simultaneous \
              --save_iter $save_iter \
              --print_iter $print_iter
