"""Fully-connected neural network example using MNIST dataset

This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
import argparse
import numpy as np

import chainer
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers

from train import MLP


n_class = 10


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)


def compute_large_margin_loss(model, t, n_class, gamma=1, eps=1e-6):
    out = softmax(model.outputs[-1].data)
    t_one_hot = np.zeros_like(out)
    t_one_hot[np.arange(len(t)), t] = 1

    diff_out = out - t_one_hot
    loss = None
    for h in model.outputs:
        res = gamma+diff_out/(np.linalg.norm(eps+h.grad, axis=1)).reshape(-1, 1)
        res = np.where(res < 0, 0., res).astype(np.float32)
        res = res.sum(axis=1)
        if loss is None:
            loss = res
        else:
            loss += res
    return loss.mean()


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot using model '
                             'and state files in the specified directory')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    model = MLP(args.unit, n_class)
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))

    if args.resume:
        # Resume from a snapshot
        serializers.load_npz('{}/mlp.model'.format(args.resume), model)
        serializers.load_npz('{}/mlp.state'.format(args.resume), optimizer)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_count = len(train)
    test_count = len(test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0

    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        # optimizer.update(model, x, t)

        loss = model(x, t)
        loss.backward(retain_grad=True)
        model.loss.data = \
            compute_large_margin_loss(model, t_array, n_class, gamma=1, eps=1e-6)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

        if train_iter.is_new_epoch:
            print('epoch: ', train_iter.epoch)
            print('train mean loss: {}, accuracy: {}'.format(
                sum_loss / train_count, sum_accuracy / train_count))
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            for batch in test_iter:
                x_array, t_array = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            test_iter.reset()
            print('test mean  loss: {}, accuracy: {}'.format(
                sum_loss / test_count, sum_accuracy / test_count))
            sum_accuracy = 0
            sum_loss = 0

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('{}/mlp.model'.format(args.out), model)
    print('save the optimizer')
    serializers.save_npz('{}/mlp.state'.format(args.out), optimizer)


if __name__ == '__main__':
    main()
