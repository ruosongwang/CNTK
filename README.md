# Convolutional Neural Tangent Kernel (CNTK)

This repository contains the code for Convolutional Neural Tangent Kernel (CNTK) in the following paper 

[On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/abs/1904.11955) (NeurIPS 2019)

### Citation

	@inproceedings{arora2019exact,
	  title={On exact computation with an infinitely wide neural net},
	  author={Arora, Sanjeev and Du, Simon S. and Hu, Wei and Li, Zhiyuan and Salakhutdinov, Ruslan and Wang, Ruosong},
	  booktitle={Thirty-third Conference on Neural Information Processing Systems},
	  year={2019}
	}
	
## Usage
Require Python 2.7 and CUDA.

1. Install [CuPy](https://cupy.chainer.org).
2. Download CIFAR-10.
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz; tar zxvf cifar-10-python.tar.gz
```
3. Parallelize [Line 143-146](https://github.com/ruosongwang/CNTK/blob/f6152dab94dfc7abb84cba8eb346366d8c39c0f0/CNTK.py#L143) in CNTK.py according to your specific computing enviroment to utilize multiple GPUs. 

To reproduce results in Table 1 in our paper:

For column CNTK-V:

```
python CNTK.py --gap no --fix no --depth DEPTH
```
where DEPTH is 3, 4, 6, 11 or 21.

For column CNTK-GAP:

```
python CNTK.py --gap yes --fix yes --depth DEPTH
```
where DEPTH is 3, 4, 6, 11 or 21.

