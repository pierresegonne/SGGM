import math
import numpy as np

from sggm.baselines.utils import ds_from_dl


def gp(args, dm):
    X, y = ds_from_dl(dm.train_dataloader(), "cpu")
    X, y = X.numpy(), y.numpy()
    Xval, yval = ds_from_dl(dm.test_dataloader(), "cpu")
    Xval, yval = Xval.numpy(), yval.numpy()
    if X.shape[0] > 2000:  # do not run gp for large datasets
        return np.nan, np.nan
    import GPy

    d = X.shape[1]
    kernel = GPy.kern.RBF(d, ARD=True)
    model = GPy.models.GPRegression(X, y.reshape(-1, 1), kernel, normalizer=True)

    model.constrain_positive(" ")  # ensure positive hyperparameters
    model.optimize()

    y_pred, cov = model.predict(Xval, full_cov=True)
    cov += 1e-4 * np.diag(np.ones(cov.shape[0]))
    y_pred = y_pred.flatten()
    log_px = (
        -1
        / 2
        * (
            np.linalg.slogdet(cov)[1]
            + (yval - y_pred).T.dot(np.linalg.inv(cov).dot(yval - y_pred))
            + d * math.log(2 * math.pi)
        )
        / Xval.shape[0]
    )

    rmse = math.sqrt(((yval - y_pred) ** 2).mean())
    return log_px, rmse
