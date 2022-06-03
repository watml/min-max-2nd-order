#!/bin/bash

# configuration for estimation the covariance of a gaussian distribution

epoch=2000
dataset=covariance
train_size=10000
batch_size=10000

cg_maxiter=16

save_iter=1
print_iter=1

# 2000 epoch 215s
# # 2ts-gda
# d_optim=gd
# d_step_size=0.2
# d_num_step=1
# g_optim=gd
# g_step_size=0.02
# simultaneous=0

# 1462 epoch 215s
# gda-20
d_optim=gd
d_step_size=0.02
d_num_step=20
g_optim=gd
g_step_size=0.02
simultaneous=0


# # sd 
# d_optim=gd
# d_step_size=0.2
# d_num_step=1
# g_optim=sd
# g_step_size=0.02
# simultaneous=1

# # fr 
# d_optim=fr
# d_step_size=0.2
# d_num_step=1
# g_optim=gd
# g_step_size=0.02
# simultaneous=1

# # gd-newton
# d_optim=newton
# d_step_size=1.
# d_num_step=1
# g_optim=gd
# g_step_size=0.02
# simultaneous=0

# # complete newton
# d_optim=newton
# d_step_size=1.
# d_num_step=1
# g_optim=newton
# g_step_size=1.
# simultaneous=0

# # EG
d_optim=eg
d_step_size=0.02
d_num_step=1
g_optim=eg
g_step_size=0.02
simultaneous=1

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
              --cg_tol 1e-30 \
              --cg_lam 0. \
              --cg_lam_cn 0. \
              --simultaneous $simultaneous \
              --save_iter $save_iter \
              --print_iter $print_iter
