import math
import numpy as np
import torch

from itertools import chain
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from torch import distributions as D

from sggm.baselines.utils import batchify, dist, ds_from_dl


def gen_Qw(data, psu, ssu, M):
    """ Takes data as numpy matrix """
    T = KDTree(data)
    _, Q = T.query(data, k=M)
    N = Q.shape[0]
    w = psu * np.bincount(Q.flatten()) / N * ssu / M

    return (Q, w)


def locality_sampler2(psu, ssu, Q, w):
    ps = np.random.randint(len(w), size=psu)
    ss = []

    for i in ps:
        ss = np.append(ss, np.random.choice(Q[i, :], size=ssu, replace=False))
    ss = np.unique(ss)
    return ss


def t_likelihood(x, mean, var, w=None):
    def logmeanexp(inputs, dim=0):
        input_max = inputs.max(dim=dim)[0]
        return (inputs - input_max).exp().mean(dim=dim).log() + input_max

    w = torch.ones(x.shape[0], device=x.device) if w is None else w

    c = -0.5 * math.log(2 * math.pi)
    A = logmeanexp(c - var.log() / 2 - (x - mean) ** 2 / (2 * var), dim=0)
    # mean shape : [batch,dim], var shape [num,draws,batch,dim]
    # shape [batch, dim]
    A = A / w.reshape(-1, 1)
    # mean over batch size with inclusion prob if given
    # mean over dim
    A = A.sum()

    return A


def john(args, dm):

    device = args.device
    n_neurons = 50
    X, y = ds_from_dl(dm.train_dataloader(), device)
    Xval, yval = ds_from_dl(dm.train_dataloader(), device)

    args.n_clusters = min(args.n_clusters, X.shape[0])

    mean_psu = 1
    mean_ssu = 40
    mean_M = 50

    var_psu = 2
    var_ssu = 10
    var_M = 10

    num_draws_train = 20
    kmeans = KMeans(n_clusters=args.n_clusters)
    kmeans.fit(np.concatenate([X.cpu()], axis=0))
    c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    c.to(device)

    class translatedSigmoid(torch.nn.Module):
        def __init__(self):
            super(translatedSigmoid, self).__init__()
            self.beta = torch.nn.Parameter(torch.tensor([1.5]))

        def forward(self, x):
            beta = torch.nn.functional.softplus(self.beta)
            alpha = -beta * (6.9077542789816375)
            return torch.sigmoid((x + alpha) / beta)

    class GPNNModel(torch.nn.Module):
        def __init__(self):
            super(GPNNModel, self).__init__()
            self.mean = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], n_neurons),
                torch.nn.ReLU(),
                torch.nn.Linear(n_neurons, 1),
            )
            self.alph = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], n_neurons),
                torch.nn.ReLU(),
                torch.nn.Linear(n_neurons, 1),
                torch.nn.Softplus(),
            )
            self.bet = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], n_neurons),
                torch.nn.ReLU(),
                torch.nn.Linear(n_neurons, 1),
                torch.nn.Softplus(),
            )
            self.trans = translatedSigmoid()

        def forward(self, x, switch):
            d = dist(x, c)
            d_min = d.min(dim=1, keepdim=True)[0]
            s = self.trans(d_min)
            mean = self.mean(x)
            if switch:
                a = self.alph(x)
                b = self.bet(x)
                gamma_dist = D.Gamma(a + 1e-8, 1.0 / (b + 1e-8))
                if self.training:
                    samples_var = gamma_dist.rsample(torch.Size([num_draws_train]))
                    x_var = 1.0 / (samples_var + 1e-8)
                else:
                    samples_var = gamma_dist.rsample(torch.Size([1000]))
                    x_var = 1.0 / (samples_var + 1e-8)
                var = (1 - s) * x_var + s * torch.ones_like(x_var).type_as(x_var)

            else:
                var = 0.05 * torch.ones_like(mean)
            return mean, var

    model = GPNNModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.mean.parameters(), lr=1e-2)
    optimizer2 = torch.optim.Adam(
        chain(
            model.alph.parameters(), model.bet.parameters(), model.trans.parameters()
        ),
        lr=1e-4,
    )
    mean_Q, mean_w = gen_Qw(X, mean_psu, mean_ssu, mean_M)

    if X.shape[0] > 100000 and X.shape[1] > 10:
        pca = PCA(n_components=0.5)
        temp = pca.fit_transform(X)
        var_Q, var_w = gen_Qw(temp, var_psu, var_ssu, var_M)
    else:
        var_Q, var_w = gen_Qw(X, var_psu, var_ssu, var_M)

    # mean_pseupoch = get_pseupoch(mean_w,0.5)
    # var_pseupoch = get_pseupoch(var_w,0.5)
    opt_switch = 1
    mean_w = torch.tensor(mean_w).to(torch.float32).to(device)
    var_w = torch.tensor(var_w).to(torch.float32).to(device)
    model.train()

    batches = batchify(X, y, batch_size=args.batch_size)

    it = 0
    while it < args.iters:
        switch = 1.0 if it > args.iters / 2.0 else 0.0

        if it % 11:
            opt_switch = opt_switch + 1  # change between var and mean optimizer
        with torch.autograd.detect_anomaly():
            data, label = next(batches)
            if not switch:
                optimizer.zero_grad()
                m, v = model(data, switch)
                loss = (
                    -t_likelihood(label.reshape(-1, 1), m, v.reshape(1, -1, 1))
                    / X.shape[0]
                )
                loss.backward()
                optimizer.step()
            else:
                if opt_switch % 2 == 0:
                    # for b in range(mean_pseupoch):
                    optimizer.zero_grad()
                    batch = locality_sampler2(mean_psu, mean_ssu, mean_Q, mean_w)
                    m, v = model(X[batch], switch)
                    loss = (
                        -t_likelihood(y[batch].reshape(-1, 1), m, v, mean_w[batch])
                        / X.shape[0]
                    )
                    loss.backward()
                    optimizer.step()
                else:
                    # for b in range(var_pseupoch):
                    optimizer2.zero_grad()
                    batch = locality_sampler2(var_psu, var_ssu, var_Q, var_w)
                    m, v = model(X[batch], switch)
                    loss = (
                        -t_likelihood(y[batch].reshape(-1, 1), m, v, var_w[batch])
                        / X.shape[0]
                    )
                    loss.backward()
                    optimizer2.step()

        if it % 500 == 0:
            m, v = model(data, switch)
            loss = -(
                -v.log() / 2 - ((m.flatten() - label.flatten()) ** 2).reshape(1, -1, 1) / (2 * v)
            ).mean()
            print("Iter {0}/{1}, Loss {2}".format(it, args.iters, loss.item()))
        it += 1

    model.eval()

    data = Xval
    label = yval
    with torch.no_grad():
        m, v = model(data, switch)
    # m = m * y_std + y_mean
    # v = v * y_std ** 2
    # log_px = normal_log_prob(label, m, v).mean(dim=0) # check for correctness
    log_px = t_likelihood(label.reshape(-1, 1), m, v) / Xval.shape[0]  # check
    rmse = ((label - m.flatten()) ** 2).mean().sqrt()
    return log_px.mean().item(), rmse.item()
