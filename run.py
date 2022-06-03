import os

import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

from model import *

from data import *

from utils import *


def loss_wasserstein(discriminator, generator, real_data, noise):
    real_scores = discriminator(real_data)
    fake_scores = discriminator(generator(noise))

    loss = real_scores.mean() - fake_scores.mean()
    for param in discriminator.parameters():
        loss = loss - 1e-8 * torch.sum(param ** 2)

    return loss


def loss_js(discriminator, generator, real_data, noise):
    ones = torch.ones((real_data.size(0), 1), dtype=torch.float, device=device).double()
    zeros = torch.zeros((noise.size(0), 1), dtype=torch.float, device=device).double()

    real_scores = discriminator(real_data)
    fake_scores = discriminator(generator(noise))

    loss = - F.binary_cross_entropy_with_logits(real_scores, ones) - F.binary_cross_entropy_with_logits(fake_scores, zeros)

    if args.dataset == 'covariance':
        for param in discriminator.parameters():
            loss = loss - 1e-5 * torch.sum(param ** 2)
            # loss = loss - 0. * torch.sum(param ** 2)

    return loss


def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]


def hxx_product(loss_fn, discriminator, generator, tensors):
    d_generator = autograd(loss_fn(), generator.parameters(), create_graph=True)
    return autograd(dot(d_generator, tensors), generator.parameters())


def hyy_product(loss_fn, discriminator, generator, tensors):
    d_discriminator = autograd(loss_fn(), discriminator.parameters(), create_graph=True)
    return autograd(dot(d_discriminator, tensors), discriminator.parameters())


def hyx_product(loss_fn, discriminator, generator, tensors):
    d_generator = autograd(loss_fn(), generator.parameters(), create_graph=True)
    return autograd(dot(d_generator, tensors), discriminator.parameters())


def hxy_product(loss_fn, discriminator, generator, tensors):
    d_discriminator = autograd(loss_fn(), discriminator.parameters(), create_graph=True)
    return autograd(dot(d_discriminator, tensors), generator.parameters())


def hfull_product(loss_fn, discriminator, generator, tensors):
    d_generator_discriminator = autograd(loss_fn(), concat(generator.parameters(), discriminator.parameters()), create_graph=True)
    return autograd(dot(d_generator_discriminator, tensors), concat(generator.parameters(), discriminator.parameters()))


"""Deprecated hessian-vector product. Does not handle None gradient."""
# def hxx_product(loss_fn, discriminator, generator, tensors):
#     d_generator = autograd.grad(loss_fn(), generator.parameters(), create_graph=True)
#     return autograd.grad(dot(d_generator, tensors), generator.parameters())
#
#
# def hyy_product(loss_fn, discriminator, generator, tensors):
#     d_discriminator = autograd.grad(loss_fn(), discriminator.parameters(), create_graph=True)
#     return autograd.grad(dot(d_discriminator, tensors), discriminator.parameters(), allow_unused=True)
#
#
# def hyx_product(loss_fn, discriminator, generator, tensors):
#     d_generator = autograd.grad(loss_fn(), generator.parameters(), create_graph=True)
#     return autograd.grad(dot(d_generator, tensors), discriminator.parameters())
#
#
# def hxy_product(loss_fn, discriminator, generator, tensors):
#     d_discriminator = autograd.grad(loss_fn(), discriminator.parameters(), create_graph=True)
#     return autograd.grad(dot(d_discriminator, tensors), generator.parameters())
#
#
# def hfull_product(loss_fn, discriminator, generator, tensors):
#     d_generator_discriminator = autograd.grad(loss_fn(), concat(generator.parameters(), discriminator.parameters()), create_graph=True)
#     return autograd.grad(dot(d_generator_discriminator, tensors), concat(generator.parameters(), discriminator.parameters()))


def get_g_update(loss_fn, discriminator, generator,
                 d_optim, d_step_size, g_optim, g_step_size,
                 cg_maxiter = 0, cg_maxiter_cn = 0, cg_tol = 0, cg_lam = 0, cg_lam_cn = 0):
    """Compute the update on generator to be added to the parameters"""
    d_generator = autograd(loss_fn(), generator.parameters())

    if g_optim == "gd" or g_optim == 'eg':
        return [- g_step_size * xx for xx in d_generator]

    if g_optim == "sd":
        inv_hyy_dy = conjugate_gradient(lambda tensors: hyy_product(loss_fn, discriminator, generator, tensors=tensors),
                                        autograd(loss_fn(), discriminator.parameters()),
                                        maxiter=cg_maxiter,
                                        tol=cg_tol,
                                        lam=cg_lam,
                                        )
        hxy_inv_hyy_dy = hxy_product(loss_fn, discriminator, generator, inv_hyy_dy)

        return [- g_step_size * xx + g_step_size * yy for xx, yy in zip(d_generator, hxy_inv_hyy_dy)]

    elif g_optim == "newton":
        """
        CAUTION: inv_hessian_dx is of size # of parameters in generator and discriminator,
                 but the dimension works out in zip update outside this function anyway.
        """
        # The default choice of cg_maxiter_cn is the same as cg_maxiter
        if cg_maxiter_cn == 0:
            cg_maxiter_cn = cg_maxiter

        #print("decaying reg: ", cg_lam_cn/i)
        inv_hessian_dx = conjugate_gradient(lambda tensors: hfull_product(loss_fn, discriminator, generator, tensors),
                                            concat(autograd(loss_fn(), generator.parameters()), [xx.new_zeros(xx.size()) for xx in discriminator.parameters()]),
                                            maxiter=cg_maxiter_cn,
                                            tol=cg_tol,
                                            lam=cg_lam_cn,
                                            )

        return [- g_step_size * xx for xx in inv_hessian_dx]


def get_d_update(loss_fn, discriminator, generator,
                 d_optim, d_step_size, g_optim, g_step_size,
                 cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn, i):
    """Compute the update on discriminator to be added to the parameters"""
    d_discriminator = autograd(loss_fn(), discriminator.parameters())
    #lambda_max = largest_eig(lambda u: hyy_product(loss_fn, discriminator, generator, tensors=u), discriminator.parameters())
    #lambda_max = largest_eig(lambda u: hxx_product(loss_fn, discriminator, generator, tensors=u), generator.parameters())
    #print("epoch: ", i, " largest eigenvalue: ", lambda_max)
    #largest_eigs[i - 1] = lambda_max
    #lambda_min = smallest_eig(lambda u: hyy_product(loss_fn, discriminator, generator, tensors=u), discriminator.parameters())
    #smallest_eigs[i - 1] = lambda_min
    #print("smallest eigenvalue hyy: ", lambda_min)
    #print("smallest eigenvalue hxx: ", smallest_eig(lambda u: hxx_product(loss_fn, discriminator, generator, tensors=u), generator.parameters()))

    if d_optim == "gd" or d_optim == "eg":
        return [d_step_size * xx for xx in d_discriminator]

    elif d_optim == "fr":
        autograd(loss_fn(), generator.parameters())
        inv_hyy_hyx_dx = conjugate_gradient(lambda tensors: hyy_product(loss_fn, discriminator, generator, tensors=tensors),
                                            hyx_product(loss_fn, discriminator, generator,
                                                        autograd(loss_fn(), generator.parameters())
                                                        ),
                                            maxiter=cg_maxiter,
                                            tol=cg_tol,
                                            lam=cg_lam,
                                            )

        return [d_step_size * xx + g_step_size * yy for xx, yy in zip(d_discriminator, inv_hyy_hyx_dx)]

    elif d_optim == "newton":
        inv_hyy_dy = conjugate_gradient(lambda tensors: hyy_product(loss_fn, discriminator, generator, tensors=tensors),
                                        d_discriminator,
                                        maxiter=cg_maxiter,
                                        tol=cg_tol,
                                        lam=cg_lam,
                                        )
        return [- d_step_size * xx for xx in inv_hyy_dy]


def eigenvalue(loss_fn, discriminator, generator, hvp):
    """Compute the eigenvalues and form the of Hessian. Only applicable for the covariance problem."""
    m, n = discriminator.W.size()
    hyy = torch.tensor([], device=device, dtype=torch.float64)
    """Create standard basis for the discriminator and multiply with hyy"""
    for k in range(m):
        for j in range(n):
            tensor = torch.zeros([m, n], device=device, dtype=torch.float64, requires_grad=False)
            tensor[k, j] += 1.
            column = torch.flatten(hvp(loss_fn, discriminator, generator, [tensor])[0])
            column = torch.unsqueeze(column, dim=0)
            hyy = torch.cat((hyy, column))
    hyy = hyy.to('cpu').numpy()
    print(hyy.shape)
    eig = np.linalg.eigvals(hyy)
    return eig, hyy


def train(discriminator, generator, loader, noise_generator, device="cuda", epoch=1,
          d_optim="gd", d_step_size=0.02, d_num_step=1,
          g_optim="gd", g_step_size=0.01,
          cg_maxiter=None, cg_maxiter_cn=None, cg_tol=None, cg_lam=None, cg_lam_cn=None,
          simultaneous=False, line_search=False,
          save_folder=None, save_iter=2, print_iter=2):

    if d_optim == "adam":
        _d_optim = optim.Adam(discriminator.parameters(), lr=d_step_size)

    elif d_optim == "amsgrad":
        _d_optim = optim.Adam(discriminator.parameters(), lr=d_step_size, amsgrad=True)

    elif d_optim == "rmsprop":
        _d_optim = optim.RMSprop(discriminator.parameters(), lr=d_step_size)


    if g_optim == "adam":
        _g_optim = optim.Adam(generator.parameters(), lr=g_step_size)

    elif g_optim == "amsgrad":
        _g_optim = optim.Adam(generator.parameters(), lr=g_step_size, amsgrad=True)

    elif g_optim == "rmsprop":
        _g_optim = optim.RMSprop(generator.parameters(), lr=g_step_size)

    limit = 215.
    start_time = time.time()
    time_seq = []

    for i in range(1, epoch + 1):
        cur_time = time.time() - start_time
        print("cur time: ", cur_time)
        time_seq.append(cur_time)
        if i == epoch:
            print("{:f} seconds in {:d} epochs".format(time.time() - start_time, epoch))

        for batch_idx, data in enumerate(loader):
            real_data = data[0].to(device)
            noise = noise_generator(real_data, generator)

            def loss_fn():
                if args.dataset == "mnist":
                    return loss_wasserstein(discriminator, generator, real_data, noise)
                else:
                    return loss_js(discriminator, generator, real_data, noise)

            def loss_fn_eg():
                if args.dataset == "mnist":
                    return loss_wasserstein(dis_half, gen_half, real_data, noise)
                else:
                    return loss_js(dis_half, gen_half, real_data, noise)

            def loss_fn_ls(discriminator, generator):
                if args.dataset == "mnist":
                    return loss_wasserstein(discriminator, generator, real_data, noise)
                else:
                    return loss_js(discriminator, generator, real_data, noise)

            def loss_fn_full_batch():
                total_loss = torch.tensor([0.], dtype=torch.double, device=device)

                for batch_idx, data in enumerate(loader):
                    real_data = data[0].to(device)
                    noise = noise_generator(real_data, generator)
                    total_loss = total_loss + loss_wasserstein(discriminator, generator, real_data, noise)

                return total_loss / (batch_idx + 1)

            if simultaneous:
                g_update = get_g_update(loss_fn, discriminator, generator,
                                        d_optim, d_step_size, g_optim, g_step_size,
                                        cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn)

                d_update = get_d_update(loss_fn, discriminator, generator,
                                        d_optim, d_step_size, g_optim, g_step_size,
                                        cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn, i)
                
                if g_optim == "eg" and d_optim == "eg":
                    dis_half = copy.deepcopy(discriminator)
                    gen_half = copy.deepcopy(generator)
                    with torch.no_grad():
                        for param, update in zip(gen_half.parameters(), g_update):
                            param += update

                    with torch.no_grad():
                        for param, update in zip(dis_half.parameters(), d_update):
                            param += update


                    g_update = get_g_update(loss_fn_eg, dis_half, gen_half,
                                        d_optim, d_step_size, g_optim, g_step_size,
                                        cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn)

                    d_update = get_d_update(loss_fn_eg, dis_half, gen_half,
                                        d_optim, d_step_size, g_optim, g_step_size,
                                        cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn, i)

                with torch.no_grad():
                    for param, update in zip(generator.parameters(), g_update):
                        param += update

                with torch.no_grad():
                    for param, update in zip(discriminator.parameters(), d_update):
                        param += update

            else:
                """Update generator"""
                if g_optim in ["adam", "amsgrad", "rmsprop"]:
                    _g_optim.zero_grad()
                    loss = loss_fn()
                    loss.backward()
                    _g_optim.step()

                else:
                    g_update = get_g_update(loss_fn, discriminator, generator,
                                            d_optim, d_step_size, g_optim, g_step_size,
                                            cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn)

                    with torch.no_grad():
                        for param, update in zip(generator.parameters(), g_update):
                            param += update

                """Update discriminator"""
                if d_optim in ["adam", "amsgrad", "rmsprop"]:
                    for j in range(10):
                        _d_optim.zero_grad()
                        loss = - loss_fn()
                        loss.backward()
                        _d_optim.step()

                else:
                    for _ in range(d_num_step):
                        d_update = get_d_update(loss_fn, discriminator, generator,
                                                d_optim, d_step_size, g_optim, g_step_size,
                                                cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn, i)

                        with torch.no_grad():
                            for param, update in zip(discriminator.parameters(), d_update):
                                param += update

                    if line_search and False:
                        alpha, rho, const = 1.0, 0.8, 0.5
                        alpha = 1. / norm(d_update)
                        print("grad_dot_pk: ", dot(autograd(loss_fn(), discriminator.parameters()), d_update))
                        while True:
                            # print("alpha: ", alpha)
                            dis_new = copy.deepcopy(discriminator)
                            # dis_new.parameters() = add(discriminator.parameters(), multi(d_update, alpha))
                            for param_new, param, update in zip(dis_new.parameters(), discriminator.parameters(), d_update):
                                param_new.data = param + alpha * update
                            # for param_new, param in zip(dis_new.parameters(), discriminator.parameters()):
                                # print("diff of params: ", param_new - param)
                            if loss_fn_ls(dis_new, generator) >= loss_fn_ls(discriminator, generator) + const * alpha * abs(dot(autograd(loss_fn(), discriminator.parameters()), d_update)):
                                break
                            else:
                                alpha = alpha * rho
                        print("final alpha: {:3f}".format(alpha))
                        print("final improvement of fn: ", loss_fn_ls(dis_new, generator) - loss_fn_ls(discriminator, generator))
                        d_update = multi(d_update, alpha)

                        for param, update in zip(discriminator.parameters(), d_update):
                            param.data += update

                        if i == epoch and args.dataset == 'covariance':
                            print("final epoch:")
                            eig, hyy = eigenvalue(loss_fn, discriminator, generator, hyy_product)
                            eig2, hxx = eigenvalue(loss_fn, discriminator, generator, hxx_product)
                            eig3, hxy = eigenvalue(loss_fn, discriminator, generator, hxy_product)
                            print("eig hyy: ", eig)
                            print("hyy: ", hyy)
                            print("hxx: ", hxx)
                            print('eig hxx: ', np.linalg.eigvals(hxx))
                            print("hxy: ", hxy)
                            schur = hxx - np.matmul(hxy, np.matmul(np.linalg.pinv(hyy), np.transpose(hxy)))
                            print('schur: ', schur)
                            print('eig schur: ', np.linalg.eigvals(schur))
                            print("condition number: ", np.amax(-eig) / np.amin(-eig))
                            # compute the eigenvalues of the single gaussian problem
                            # tensors1 = torch.tensor([[1.0],[0.0]], device='cuda:0', dtype=torch.float64, requires_grad=False)
                            # tensors2 = torch.tensor([[0.0],[1.0]], device='cuda:0', dtype=torch.float64, requires_grad=False)
                            # first_column = torch.flatten(hyy_product(loss_fn, discriminator, generator, [tensors1])[0])
                            # second_column = torch.flatten(hyy_product(loss_fn, discriminator, generator, [tensors2])[0])
                            # hyy = torch.stack((first_column, second_column), 0).to('cpu').numpy()
                            # eig = np.linalg.eig(hyy)[0]
                            # print(hyy)
                            # print(eig)

        if i % save_iter == 0 and save_folder is not None:
            fnorm = torch.tensor([0.])
            if args.dataset == 'covariance':
                emp_sigma = real_data.t().mm(real_data) / real_data.size(0)
                v = generator.V.clone().detach()
                emp_vvt = v.t().mm(noise.t().mm(noise) / noise.size(0)).mm(v)
                diff = emp_sigma - emp_vvt
                diff = [diff]
                fnorm = norm(diff)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

        if i % print_iter == 0:
            if args.dataset == "mnist":
                print("epoch: {:4d}".format(i),
                      "real predict: {:.8e}".format(discriminator(real_data).mean()),
                      "fake predict: {:.8e}".format(discriminator(generator(noise)).mean()),
                      "d_discriminator: {:20.18e}".format(norm(autograd(loss_fn_full_batch(), discriminator.parameters()))),
                      "d_generator: {:20.18e}".format(norm(autograd(loss_fn_full_batch(), generator.parameters()))),
                      )
            else:
                print("epoch: {:4d}".format(i),
                      "real predict: {:.8e}".format(discriminator(real_data).sigmoid().mean()),
                      "fake predict: {:.8e}".format(discriminator(generator(noise)).sigmoid().mean()),
                      "d_discriminator: {:20.18e}".format(norm(autograd(loss_fn_full_batch(), discriminator.parameters()))),
                      "d_generator: {:20.18e}".format(norm(autograd(loss_fn_full_batch(), generator.parameters()))),
                      "fnorm: {:.8e}".format(fnorm) if args.dataset == "covariance" else "",
                      )

            # torch.save({'model_state_dict': discriminator.state_dict(),
            #             'gradient': autograd(loss_fn_full_batch(), discriminator.parameters())
            #             },
            #            os.path.join(save_folder, "discriminator-epoch_{:d}.tar".format(i))
            #            )

            # torch.save({'model_state_dict': generator.state_dict(),
            #             'gradient': autograd(loss_fn_full_batch(), generator.parameters()),
            #             'generator_norm': fnorm if args.dataset == "covariance" else None,
            #             },
            #            os.path.join(save_folder, "generator-epoch_{:d}.tar".format(i))
            #            )

    current_time = time.time()
    print("total running time {:f}".format(current_time - start_time))


def get_save_folder(dataset, d_optim, d_step_size, d_num_step, g_optim, g_step_size):
    return "./checkpoints/{}/{}-{}-{}-{}-{}".format(dataset, g_optim, g_step_size, d_optim, d_step_size, d_num_step)


if __name__ == "__main__":
    device = "cuda:0"

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="single_gaussian", help="single_gaussian | single_gaussian_ill_conditioned")
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)

    # parser.add_argument("--pretrained_discriminator", type=lambda x: None if x == "None" else x, default=None)
    # parser.add_argument("--pretrained_generator", type=lambda x: None if x == "None" else x, default=None)
    parser.add_argument("--pretrain", type=int, default=0)
    # parser.add_argument("--save_loc_suffix", type=lambda x: "" if x == "None" else x, default="")

    parser.add_argument("--d_optim", type=str, default="gd", help="gd | newton | fr")
    parser.add_argument("--d_step_size", type=float, default=0.01)
    parser.add_argument("--d_num_step", type=int, default=1)

    parser.add_argument("--g_optim", type=str, default="gd", help="gd | newton")
    parser.add_argument("--g_step_size", type=float, default=0.01)

    parser.add_argument("--cg_maxiter", type=int, default=64)
    parser.add_argument("--cg_maxiter_cn", type=int, default=0)
    parser.add_argument("--cg_tol", type=float, default=0.01)
    parser.add_argument("--cg_lam", type=float, default=0.0)
    parser.add_argument("--cg_lam_cn", type=float, default=0.0)

    parser.add_argument("--simultaneous", type=int, default=0)
    parser.add_argument("--line_search", type=int, default=0)

    parser.add_argument("--save_iter", type=int, default=2)
    parser.add_argument("--print_iter", type=int, default=2)

    args = parser.parse_args()
    print(args)

    # largest_eigs = np.array([0.]*args.epoch)
    # smallest_eigs = np.array([0.]*args.epoch)

    set_seed(0)

    if args.dataset in ["single_gaussian", "single_gaussian_ill_conditioned"]:
        discriminator = OneLayerNet(input_dim=2).to(device).double()
        generator = ShiftNet(input_dim=2).to(device).double()

    elif args.dataset == 'covariance':
        discriminator = QuadraticNet(input_dim=2).to(device).double()
        generator = AffineNet(input_dim=2, output_dim=2).to(device).double()

    elif args.dataset == "gmm":
        discriminator = DNet().to(device).double()
        generator = GNet().to(device).double()

        if args.pretrain:
            # pattern = "./checkpoints/gmm/pretrain/{}-epoch_199.tar"
            # pattern = "./checkpoints/gmm/gd-0.01-newton-1.0-1/{}-epoch_100.tar"
            pattern = './checkpoints/gmm/_gd-gd/{}-epoch_89.tar'

            discriminator.load_state_dict(torch.load(pattern.format("discriminator"))['model_state_dict'])
            generator.load_state_dict(torch.load(pattern.format("generator"))['model_state_dict'])

    elif args.dataset == "mnist":
        discriminator = Discriminator().to(device).double()
        generator = Generator().to(device).double()

        if args.pretrain:
            pattern = "./checkpoints/mnist/gd-0.01-gd-0.01-1/{}-epoch_300.tar"

            discriminator.load_state_dict(torch.load(pattern.format("discriminator"))['model_state_dict'])
            generator.load_state_dict(torch.load(pattern.format("generator"))['model_state_dict'])

    dataset = get_data(args.dataset, args.train_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train(discriminator.train(), generator.train(), loader, NoiseGenerator(device), device=device, epoch=args.epoch,
          d_optim=args.d_optim, d_step_size=args.d_step_size, d_num_step=args.d_num_step,
          g_optim=args.g_optim, g_step_size=args.g_step_size,
          cg_maxiter=args.cg_maxiter, cg_maxiter_cn=args.cg_maxiter_cn, cg_tol=args.cg_tol, cg_lam=args.cg_lam, cg_lam_cn=args.cg_lam_cn,
          simultaneous=args.simultaneous, line_search=args.line_search,
          save_folder=get_save_folder(dataset=args.dataset,
                                      d_optim=args.d_optim, d_step_size=args.d_step_size, d_num_step=args.d_num_step,
                                      g_optim=args.g_optim, g_step_size=args.g_step_size),
          save_iter=args.save_iter, print_iter=args.print_iter,
          )
    # print("largest eigs: ", largest_eigs)
    # np.save('eigs/'+ args.dataset + "/largest_eig.npy", largest_eigs)
    # print("smallest eigs: ", smallest_eigs)
    # np.save('eigs/'+ args.dataset + "/smallest_eig.npy", smallest_eigs)
