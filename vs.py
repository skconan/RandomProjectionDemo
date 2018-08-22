# (c) 2012 Andreas Mueller amueller@ais.uni-bonn.de
# License: BSD 2-Clause
#
# See my blog for details: http://peekaboo-vision.blogspot.com

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class RandomProjector(object):
    def __init__(self, X, y, frames=200):
        self.X = X
        self.y = y
        self.frames = frames

        num_classes = len(np.unique(y))
        colors = ['r','g','b','o','y','lightgreen','cyan','pink','violet','brown']
        

        self.points = [plt.plot([], [], 'o', color=colors[i])[0]
                  for i in range(num_classes)]

        n_features = X.shape[1]

        # initialize projection matrix
        self.projection = np.zeros((n_features, 2))

        # rest is n_features - 2 large
        size = n_features - 2
        self.frequencies = 10 + np.random.randint(10, size=(size, 2))
        self.phases = np.random.uniform(size=(size, 2))

    def init_figure(self):
        for p in self.points:
            p.set_data([], [])
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))
        plt.xticks(())
        plt.yticks(())
        return self.points

    def animate(self, i):
        # set top 2x2 to identity
        self.projection[0, 0] = 1
        self.projection[1, 1] = 1
        # set "free entries" of projection matrix
        # gives them a "rotation" feel and makes the whole thing seamless.
        scale = 2 * np.pi * i / self.frames
        self.projection[2:, :] = np.sin(self.frequencies * scale + self.phases)
        interpolation = np.dot(X, self.projection)
        interpolation /= interpolation.max(axis=0)
        for p, c in zip(self.points, np.unique(y)):
            p.set_data(interpolation[y == c, 0], interpolation[y == c, 1])
        return self.points


def make_video(X, y=None, frames=500, filename="video.mp4"):
    fig = plt.figure()
    projector = RandomProjector(X, y, frames)
    anim = FuncAnimation(fig, projector.animate, frames=frames, interval=100,
            blit=True, init_func=projector.init_figure)
    plt.show()

if __name__ == "__main__":
    data = load_digits()
    X, y = data.data, data.target

    mask = (y == 1) + (y == 4) + (y == 7)
    y = y[mask]
    X = X[mask]

    # we should at least remove the mean
    X = StandardScaler(with_std=False).fit_transform(X)

    # make boring PCA visualization for comparison
    num_classes = len(np.unique(y))
    colors = ['r','g','b','o','y','lightgreen','cyan','pink','violet','brown']

    # X_pca = PCA(n_components=2).fit_transform(X)
    # for i, c in enumerate(np.unique(y)):
    #     plt.plot(X_pca[y == c, 0], X_pca[y == c, 1], 'o', color=colors[i],
    #             label=c)
    # plt.legend()
    # plt.savefig("digits_pca.png", bbox_inches="tight")
    # PCA here optional. Also try without.
    X = PCA().fit_transform(X)
    # plt.show()
    make_video(X, y, filename='digits_two_classes.mp4', frames=1000)
