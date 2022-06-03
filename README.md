# Second order methods for minimax optimization 

This repo includes code for running first- and second-order methods for minimax optimization, including gradient descent ascent (GDA), total gradient descent ascent (TGDA), follow-the-ridge (FR), gradient-descent Newton (GDN) and complete Newton (CN). We run these algorithms on tasks including estimation of a single Gaussian, learning a mixture of Gaussians, and generating digits on the MNIST 0/1 subset. GPU/CUDA required for running all our experiments. 

* `model.py` contains different neural net architectures for various tasks
* `data.py` generates various datasets
* `run.py` is the main python script for comparing different algorithms
* `utils.py` implements various helper functions including hessian-vector-product and conjugate gradient.

## Scripts for running the experiments
The following bash files include configurations to run various experiments. You can uncomment different parts to run different algorithms including GDA/TGDA/FR/GDN/CN.

* `bash_gaussian_mean.sh`: estimation of the mean of a Gaussian
* `bash_gaussian_covariance.sh`: estimation of the covariance of a Gaussian
* `bash_gmm.sh`: learning a mixture of Gaussians
* `bash_gmm.sh`: learning to generate digits in MNIST 0/1 dataset

## Pretrained models
The folder `./checkpoints` includes two pretrained models for GMM and MNIST separately. We used the model trained by GDA as initialization.

## Plot files
* `plot_gmm.py`: plot file for visualizing the Gaussian mixtures generated by different algorithms
* `plot_mnist.py`: plot file for visualizing the digits generated generated by different algorithms
