import numpy as np
import matplotlib.pyplot as plt


def scatter_mag(x, y, mag, **kwargs):
    flux = 10**(mag / (-2.5))
    size = flux**0.5
    size = size / size.mean() * 3
    plt.scatter(x, y, s=size, **kwargs)


def cmd(col, mag, xlim=None, ylim=None,
        name='Open Cluster', col_name='bp-rp', dpi=100):
    flux = 10**(mag / (-2.5))
    size = flux**0.5
    size = size / size.mean() * 3

    rad = size * 5
    c = plt.cm.rainbow((col - col.min()) / (col.max() - col.min()))
    facecolor = c.copy()
    facecolor[:, -1] = 0.2
    with plt.rc_context({'xtick.color': '0.8',
                         'ytick.color': '0.8',
                         'axes.edgecolor': '0.8',
                         'text.color': 'white'}):
        plt.figure(facecolor='0.4', dpi=dpi)
        ax = plt.gca()
        ax.set_facecolor('0.4')
        plt.scatter(col, mag, marker='o',
                    s=rad, facecolor=facecolor, edgecolor=c, linewidth=0.5)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        else:
            ax.invert_yaxis()

        plt.xlabel(col_name, color='white')
        plt.ylabel("g", color='white')

        plt.text(1, -0.15, 'Gaia EDR3',
                 ha='right', va='bottom', transform=ax.transAxes, fontsize=8)

        plt.text(0.5, 1.05, f"Colour Magnitude Diagram: {name}",
                 ha='center', va='center', transform=ax.transAxes, fontsize=12)


def univariate_gaussian(x, A, mu, sig):
    """Un-normalized Gaussian"""
    return A * np.exp(-(x - mu)**2 / (2 * sig**2))


def multivariate_gaussian(X, mu, Sigma):
    """Multivariate Gaussian distribution"""
    k = mu.shape[0]

    V = np.linalg.inv(Sigma)
    norm = np.sqrt((2 * np.pi)**k * np.linalg.det(Sigma))

    res = X - mu

    return 1 / norm * np.exp(-1 / 2 * np.einsum('...j,jk,...k', res, V, res))
