# One-shot Learning with Memory-Augmented Neural Networks
Tensorflow implementation of the paper *One-shot Learning with Memory-Augmented Neural Networks*, by A. Santoro et al.
This is an offshoot of a [larger project](https://github.com/adityagilra/archibrain), which aims to synthesize bio-plausible neural networks that solve cognitive tasks.

This implementation is much simpler than a lot of others out there, thanks to TensorFlow's API and ease of use. The model as described in the paper has been followed as closely as possible.

The code is inspired by [tristandeleu](https://github.com/tristandeleu)'s excellent Theano implementation.

## Requirements
This implementation requires
* numpy >= 1.12.1
* scipy >= 0.17.0
* tensorflow >= 1.0
* matplotlib >= 1.5.1

## Dataset
All datasets should be placed in the [`data`](data) folder.
In order to run the Omniglot experiment, download the dataset from [here](https://github.com/brendenlake/omniglot) and unzip [`python/images_background.zip`] in the [`data/omniglot`](data/omniglot) folder.

## Running
`python omniglot.py`

Run `python omniglot.py --help` to see a list of available options


`--num-classes` and `--num-samples` specify how to form an episode of input data from the omniglot dataset.


`--controller-size` specifies the number of hidden units in the LSTM controller network.

`--memory-locations` specifies the number of memory locations contained in the external memory.

`--memory-word-size` specifies the size of each word stored in external memory.


`--learning-rate` and `--iterations` specify the parameters for training the model.


The default values for all parameters are those used in the paper.

## Paper
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap, *One-shot Learning with Memory-Augmented Neural Networks*, [[arXiv](http://arxiv.org/abs/1605.06065)]
