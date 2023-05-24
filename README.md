# AutoEncoderST


## Introduction


This module is an implementation of a neural network with an autoencoder architecture, with the purpose of reducing the number of parameters to describe an effective string model.

Heterotic string theory effective models with norm group are created using: the information of the orbifold geometry and the way of embedding the space in the norm group. This module needs both, however, all geometries leading to an effective theory in 4 dimensions with N=1 supersymmetry have already been classified in this [article](https://arxiv.org/abs/1209.3906). The list of geometries can be found in the file List_of_geometries.

There are many ways to embed the space group, although, it is necessary to satisfy certain equations, however, the number is huge. 
One way to search for the particular forms is to do a random search on the parameters, which turn out to be 8, 16-dimensional, vectors. This approach has already been addressed in this [article](https://arxiv.org/abs/1110.5229), using a software called The Orbifolder.


This module uses the information produced by `the orbifolder`, but before it must be transformed by another Python module called [Makedataset](http://github.com/enriqueescalante)

## Installation


For installation, you must first clone this repository or download the compressed file and unzip it.

Once located in the module folder, you just need to run

```
python setup.py install
```

to install all the necessary dependencies, for all users it is necessary to add `sudo`.

It is also important to note that there are several data files, covering Z12, Z8, among others, in the Data folder. We also provide a 
.pt file containng the weights of the (trained) neural net and can be used to reproduce the latent space plots shown in this [paper.](https://arxiv.org/abs/2212.00821)