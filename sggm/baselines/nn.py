import math
import numpy as np
import torch

from itertools import chain
from tqdm import tqdm

from sggm.baselines.utils import batchify, ds_from_dl, normal_log_prob


def nn(args, dm):
    device = args.device
    n_neurons = 50
    X, y = ds_from_dl(dm.train_dataloader(), device)
    Xval, yval = ds_from_dl(dm.train_dataloader(), device)

    mean = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], n_neurons),
        torch.nn.ReLU(),
        torch.nn.Linear(n_neurons, 1),
    )
    mean.to(device)
    var = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], n_neurons),
        torch.nn.ReLU(),
        torch.nn.Linear(n_neurons, 1),
        torch.nn.Softplus(),
    )
    var.to(device)

    optimizer = torch.optim.Adam(chain(mean.parameters(), var.parameters()), lr=args.lr)
    it = 0
    progressBar = tqdm(desc="Training nn", total=args.iters, unit="iter")
    batches = batchify(X, y, batch_size=args.batch_size)
    while it < args.iters:
        switch = 1.0 if it > args.iters / 2 else 0.0
        optimizer.zero_grad()
        data, label = next(batches)
        m, v = mean(data), var(data)
        v = switch * v + (1 - switch) * torch.tensor([0.02 ** 2], device=device)
        loss = normal_log_prob(label, m, v).sum()
        (-loss).backward()
        optimizer.step()
        it += 1
        progressBar.update()
        progressBar.set_postfix({"loss": loss.item()})
    progressBar.close()

    data = Xval
    label = yval
    m, v = mean(data), var(data)
    # Consistency with our way - only evaluate on std data.
    # m = m * y_std + y_mean
    # v = v * y_std ** 2
    log_px = normal_log_prob(label, m, v)
    rmse = ((label - m.flatten()) ** 2).mean().sqrt()
    return log_px.mean().item(), rmse.item()


def mcdnn(args, dm):
    device = args.device
    n_neurons = 50
    X, y = ds_from_dl(dm.train_dataloader(), device)
    Xval, yval = ds_from_dl(dm.train_dataloader(), device)

    mean = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], n_neurons),
        torch.nn.Dropout(p=0.05),
        torch.nn.ReLU(),
        torch.nn.Linear(n_neurons, 1),
        torch.nn.Dropout(p=0.05),
    )
    mean.to(device)

    optimizer = torch.optim.Adam(mean.parameters(), lr=args.lr)

    it = 0
    progressBar = tqdm(desc="Training nn", total=args.iters, unit="iter")
    batches = batchify(X, y, batch_size=args.batch_size)
    while it < args.iters:
        optimizer.zero_grad()
        data, label = next(batches)
        m = mean(data)
        loss = (m - label).abs().pow(2.0).mean()
        loss.backward()
        optimizer.step()
        it += 1
        progressBar.update()
        progressBar.set_postfix({"loss": loss.item()})
    progressBar.close()

    data = Xval
    label = yval
    samples = torch.zeros(Xval.shape[0], args.mcmc).to(device)
    for i in range(args.mcmc):
        samples[:, i] = mean(data).flatten()
    m, v = samples.mean(dim=1), samples.var(dim=1)

    log_probs = normal_log_prob(label, m, v)
    rmse = ((label - m.flatten()) ** 2).mean().sqrt()
    return log_probs.mean().item(), rmse.item()


def ensnn(args, dm):
    device = args.device
    n_neurons = 50
    X, y = ds_from_dl(dm.train_dataloader(), device)
    Xval, yval = ds_from_dl(dm.train_dataloader(), device)

    ms, vs = [], []
    for m in range(args.n_models):  # initialize differently
        mean = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, 1),
        )
        mean.to(device)
        var = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, 1),
            torch.nn.Softplus(),
        )
        var.to(device)

        optimizer = torch.optim.Adam(
            chain(mean.parameters(), var.parameters()), lr=args.lr
        )
        it = 0
        progressBar = tqdm(desc="Training nn", total=args.iters, unit="iter")
        batches = batchify(X, y, batch_size=args.batch_size)
        while it < args.iters:
            switch = 0.0  # 1.0 if it > args.iters/2 else 0.0
            optimizer.zero_grad()
            data, label = next(batches)
            m, v = mean(data), var(data)
            v = switch * v + (1 - switch) * torch.tensor([0.02 ** 2], device=device)
            loss = normal_log_prob(label, m, v).sum()
            (-loss).backward()
            optimizer.step()
            it += 1
            progressBar.update()
            progressBar.set_postfix({"loss": loss.item()})
        progressBar.close()

        data = Xval
        label = yval
        m, v = mean(data), var(data)
        # m = m * y_std + y_mean
        # v = v * y_std ** 2

        ms.append(m)
        vs.append(v)

    ms = torch.stack(ms)
    vs = torch.stack(vs)

    m = ms.mean(dim=0)
    v = (vs + ms ** 2).mean(dim=0) - m ** 2

    log_px = normal_log_prob(label, m, v)
    rmse = ((label - m.flatten()) ** 2).mean().sqrt()
    return log_px.mean().item(), rmse.item()


def bnn(args, dm):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability import distributions as tfd

    device = args.device
    n_neurons = 50
    X, y = ds_from_dl(dm.train_dataloader(), device)
    Xval, yval = ds_from_dl(dm.train_dataloader(), device)

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    # y, y_mean, y_std = normalize_y(y)

    def VariationalNormal(name, shape, constraint=None):
        means = tf.compat.v1.get_variable(
            name + "_mean", initializer=tf.ones([1]), constraint=constraint
        )
        stds = tf.compat.v1.get_variable(name + "_std", initializer=-1.0 * tf.ones([1]))
        return tfd.Normal(loc=means, scale=tf.nn.softplus(stds))

    x_p = tf.compat.v1.placeholder(tf.float32, shape=(None, X.shape[1]))
    y_p = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

    with tf.compat.v1.name_scope("model", values=[x_p]):
        layer1 = tfp.layers.DenseFlipout(
            units=n_neurons,
            activation="relu",
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        )
        layer2 = tfp.layers.DenseFlipout(
            units=1,
            activation="linear",
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        )
        predictions = layer2(layer1(x_p))
        noise = VariationalNormal(
            "noise", [1], constraint=tf.keras.constraints.NonNeg()
        )
        pred_distribution = tfd.Normal(loc=predictions, scale=noise.sample())

    neg_log_prob = -tf.reduce_mean(input_tensor=pred_distribution.log_prob(y_p))
    kl_div = sum(layer1.losses + layer2.losses) / X.shape[0]
    elbo_loss = neg_log_prob + kl_div

    with tf.compat.v1.name_scope("train"):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(elbo_loss)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        it = 0
        progressBar = tqdm(desc="Training BNN", total=args.iters, unit="iter")
        batches = batchify(X, y, batch_size=args.batch_size)
        while it < args.iters:
            data, label = next(batches)
            _, loss = sess.run(
                [train_op, elbo_loss], feed_dict={x_p: data.cpu(), y_p: label.reshape(-1, 1).cpu()}
            )
            progressBar.update()
            progressBar.set_postfix({"loss": loss})
            it += 1
        progressBar.close()

        W0_samples = layer1.kernel_posterior.sample(1000)
        b0_samples = layer1.bias_posterior.sample(1000)
        W1_samples = layer2.kernel_posterior.sample(1000)
        b1_samples = layer2.bias_posterior.sample(1000)
        noise_samples = noise.sample(1000)

        W0, b0, W1, b1, n = sess.run(
            [W0_samples, b0_samples, W1_samples, b1_samples, noise_samples]
        )

    def sample_net(x, W0, b0, W1, b1, n):
        h = np.maximum(np.matmul(x[np.newaxis], W0) + b0[:, np.newaxis, :], 0.0)
        return (
            np.matmul(h, W1)
            + b1[:, np.newaxis, :]
            + n[:, np.newaxis, :] * np.random.randn()
        )

    samples = sample_net(Xval, W0, b0, W1, b1, n)

    m = samples.mean(axis=0)
    v = samples.var(axis=0)

    # m = m * y_std + y_mean
    # v = v * y_std ** 2

    log_probs = normal_log_prob(yval.cpu(), m, v)
    rmse = math.sqrt(((m.flatten() - yval.cpu()) ** 2).mean())

    return log_probs.mean(), rmse
