import argparse
import os
import importlib

from train.wgan_gradient_penalty import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('data', type=str, help='data')
    parser.add_argument('model', type=str, help='model type')
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--it', type=int, default=1000000, help='# iterations')
    parser.add_argument('-b', type=int, default=4, help='burn in length')
    parser.add_argument('-m', type=int, default=3, help='eval chain length')
    parser.add_argument('-d', type=int, default=7, help='critic iterations')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    if args.gpus is not '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    dm = importlib.import_module('models.' + args.data)
    mm = importlib.import_module('models.' + args.data + '.' + args.model)

    data_sampler, noise_sampler = dm.data_sampler, dm.noise_sampler
    transition_fn, discriminator, visualizer = mm.TransitionFunction(), mm.Discriminator(), mm.visualizer

    path = 'logs/{}/{}'.format(args.data, args.model)
    try:
        os.makedirs(path)
    except Exception:
        pass

    trainer = Trainer(transition_fn, discriminator, data_sampler, noise_sampler, args.b, args.m)
    if args.load:
        trainer.load(path)
    trainer.train(
        num_batches=args.it,
        visualizer=visualizer,
        path=path,
        d_iters=args.d,
        epoch_size=mm.epoch_size,
        logging_freq=mm.logging_freq
    )
