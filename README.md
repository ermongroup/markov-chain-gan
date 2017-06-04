# Markov Chain GAN (MGAN)
Code for "Generative Adversarial Training for Markov Chains" (ICLR 2017 Workshop)

## Preprocessing
Running the code requires some preprocessing.
Namely, we transform the data is transformed to TensorFlow Records file to maximize speed 
(as [suggested by TensorFlow](https://www.tensorflow.org/performance/performance_guide)).

### MNIST
The data used for training is [here](https://drive.google.com/open?id=0B0LzoDno7qkJdDluZW5DSnpyWTg).
Download and place the directory in `~/data/mnist_tfrecords`. 

(This can be easily done by using a symlink or you can change the path in file `models/mnist/__init__.py`)

### CelebA
The data used for training is [here](https://drive.google.com/open?id=0B0LzoDno7qkJX3p2YS1DODNrM3c).
Download and place the directory in `~/data/celeba_tfrecords`.



## Running Experiments
```
python mgan.py [data] [model] -b [B] -m [M] -d [critic iterations] --gpus [gpus]
```

### MNIST
```
python mgan.py mnist mlp -b 4 -m 3 -d 7 --gpus [gpus]
```

### CelebA
```
python mgan.py celeba conv -b 4 -m 3 -d 7 --gpus [gpus]
```

### Custom Experiments
It is easy to define your own problem and run experiments.
- Create a folder `data` under the `models` directory, and define `data_sampler` and `noise_sampler` in `__init__.py`.
- Create a file `model.py` under the `models/data` directory, and define the following:
  - `class TransitionFunction(TransitionBase)` (Generator)
  - `class Discriminator(DiscriminatorBase)` (Discriminator)
  - `def visualizer(model, name)` (If you need to generate figures)
  - `epoch_size` and `logging_freq`
- That's it!
