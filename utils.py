import numpy as np
from statsmodels.stats.weightstats import _zconfint_generic
def _calc_lhat_glm(
    grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=None, clip=False
):
    """
    Calculates the optimal value of lhat for the prediction-powered confidence interval for GLMs.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        clip (bool, optional): Whether to clip the value of lhat to be non-negative. Defaults to `False`.

    Returns:
        float: Optimal value of `lhat`. Lies in [0,1].
    """
    # 实质上这个函数就是在计算lambda_hat , 先拿到三组偏导数据，对应那片论文里头的那三个loss function ， 这个函数本质上就是section 6 的那个算法
    
    
    
    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]
    cov_grads = np.zeros((d, d))

    for i in range(n): 
        cov_grads += (1 / n) * (
            np.outer(
                grads[i] - grads.mean(axis=0),
                grads_hat[i] - grads_hat.mean(axis=0),
            )
            + np.outer(
                grads_hat[i] - grads_hat.mean(axis=0),
                grads[i] - grads.mean(axis=0),
            )
        )
    var_grads_hat = np.cov(
       np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    ) 

    if coord is None:
        vhat = inv_hessian
    else:
        vhat = inv_hessian @ np.eye(d)[coord]

    if d > 1:
        num = (
            np.trace(vhat @ cov_grads @ vhat)
            if coord is None
            else vhat @ cov_grads @ vhat
        )
        denom = (
            2 * (1 + (n / N)) * np.trace(vhat @ var_grads_hat @ vhat)
            if coord is None
            else 2 * (1 + (n / N)) * vhat @ var_grads_hat @ vhat
        )
    else:
        num = vhat * cov_grads * vhat
        denom = 2 * (1 + (n / N)) * vhat * var_grads_hat * vhat

    lhat = num / denom
    if clip:
        lhat = np.clip(lhat, 0, 1)
    return lhat.item()
def ppi_mean_pointestimate(   # 做文章给的方法的均值的点估计
    Y, Yhat, Yhat_unlabeled, lhat=None, coord=None, w=None, w_unlabeled=None
):
    """Computes the prediction-powered point estimate of the d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the dimension of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set. Defaults to all ones vector.

    Returns:
        float or ndarray: Prediction-powered point estimate of the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Yhat.shape[1] if len(Yhat.shape) > 1 else 1
    w = np.ones(n) if w is None else w / w.sum() * n   # 权重为1，否则就取权重均值。
    w_unlabeled = (   # 同上
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    if lhat is None:
        ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean() + (
            w * (Y - Yhat)
        ).mean()
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
        )
        return ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return (w_unlabeled * lhat * Yhat_unlabeled).mean(axis=0) + (
            w * (Y - lhat * Yhat)
        ).mean(axis=0)
def ppi_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for a d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Y.shape[1] if len(Y.shape) > 1 else 1
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    if lhat is None:
        ppi_pointest = ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=1,
            w=w,
            w_unlabeled=w_unlabeled,
        )
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
        )
        return ppi_mean_ci(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    ppi_pointest = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    imputed_std = (w_unlabeled * (lhat * Yhat_unlabeled)).std() / np.sqrt(N)
    rectifier_std = (w * (Y - lhat * Yhat)).std() / np.sqrt(n)

    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(imputed_std**2 + rectifier_std**2),
        alpha,
        alternative,
    )
def classical_mean_ci(Y, w=None, alpha=0.1, alternative="two-sided"):
    """Classical mean confidence interval using the central limit theorem.
    Returns:
        tuple: (lower, upper) confidence interval bounds.
    """
    n = Y.shape[0]
    if w is None:
        return _zconfint_generic(
            Y.mean(), Y.std() / np.sqrt(n), alpha, alternative
        )
    else:
        w = w / w.sum() * n
        return _zconfint_generic(
            (w * Y).mean(), (w * Y).std() / np.sqrt(n), alpha, alternative
        )