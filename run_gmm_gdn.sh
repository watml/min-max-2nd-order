#!/bin/bash

# # gda

d_optim=newton
d_step_size=1.0
#d_optim=gd
#d_step_size=0.01
d_num_step=1
g_optim=gd
g_step_size=0.01
simultaneous=0

python run.py --epoch 2000 \
              --dataset gmm \
              --train_size 10000 \
              --batch_size 10000 \
              --pretrain 1 \
              --d_optim $d_optim \
              --d_step_size $d_step_size \
              --d_num_step $d_num_step \
              --g_optim $g_optim \
              --g_step_size $g_step_size \
              --cg_maxiter 20 \
              --cg_tol 1e-40 \
              --cg_lam 0.00 \
              --simultaneous $simultaneous \
              --save_iter 2 \
              --print_iter 2\
              --line_search 1\


#d_optim=newton
#d_step_size=1.0
##d_optim=gd
##d_step_size=0.02
#d_num_step=1
#g_optim=gd
#g_step_size=0.01
#simultaneous=0
#python run.py --epoch 200 \
#              --dataset mnist \
#              --train_size 10000 \
#              --batch_size 10000 \
#              --pretrain 1 \
#              --d_optim $d_optim \
#              --d_step_size $d_step_size \
#              --d_num_step $d_num_step \
#              --g_optim $g_optim \
#              --g_step_size $g_step_size \
#              --cg_maxiter 20 \
#              --cg_tol 1e-40 \
#              --cg_lam 0.0 \
#              --simultaneous $simultaneous \
#              --save_iter 2 \
#              --print_iter 2\
