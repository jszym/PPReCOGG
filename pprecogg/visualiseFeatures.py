# -*- coding: utf-8 -*-
#
# visualise.py
# ========================
# An interactive command-line interface to the
# PPReCOGG libraries.
# ========================
#
# Copyright 2017 Joseph Szymborski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
#
# visualiseFeatures.py
# ========================
# An interactive command-line interface to the
# kmKNNGabor libraries.
# ========================
#
# Copyright (c) 2017 Joseph Szymborski under the MIT License
# (see LICENSE file for details)

from sklearn.manifold import MDS
import numpy as np
import h5py

def mds_reduce(files, dimensions):

    Ys = []

    for file_num, file in enumerate(files):

        h5file = h5py.File(file, mode='r')

        features = []

        for window_name in h5file['gabor_features']:
            features.append(h5file['gabor_features'][window_name]['means'])
            features.append(h5file['gabor_features'][window_name]['stds'])

        full_features = np.array([np.array(f).flatten() for f in zip(*features)])

        full_features = full_features[:700]

        full_features = full_features.astype(np.float64)

        mds = MDS(dimensions, max_iter=100, n_init=1)

        Y = mds.fit_transform(full_features)

        Ys.append(Y)

    return Ys

def mds_reduce2(files, dimensions):

    Ys = []

    for file_num, file in enumerate(files):

        h5file = h5py.File(file, mode='r')

        features = []

        for window_name in h5file['gabor_features']:

            means = h5file['gabor_features'][window_name]['means']
            stds = h5file['gabor_features'][window_name]['stds']

            features.append(np.concatenate(means, stds))


        full_features = np.array([np.array(f).flatten() for f in zip(*features)])

        full_features = full_features[:700]

        full_features = full_features.astype(np.float64)

        mds = MDS(dimensions, max_iter=100, n_init=1)

        Y = mds.fit_transform(full_features)

        Ys.append(Y)

    return Ys

def visualise(config):
    """
    r = mds_reduce(['features/gabor_brick-wall-D94.tifff_1501135333.h5',
                'features/gabor_wood-grain-D68.tiff_1501189853.h5'], 3)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = sns.color_palette("cubehelix", 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, dataset in enumerate(r):
        ax.scatter(dataset[:,0],
                   dataset[:,1],
                   dataset[:,2],
                   c=colors[i],
                   marker="D")

    plt.show()
    """

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import seaborn as sns
    import h5py
    import numpy as np
    from sklearn.manifold import MDS, TSNE
    import os.path

    import plotly.plotly as py
    import plotly.graph_objs as go
    import plotly

    plotly.tools.set_credentials_file(username='jszymski', api_key='exozAY0cMjQdMYuHA0hU')

    files = ['features/gabor_1.3.10.tiff_s1.0_l0.25_g0.02_p0_1505497049.h5',
             'features/gabor_brick-1.3.12.tiff_s1.0_l0.25_g0.02_p0_1505497716.h5']

    files = config['known_features']

    Ys = []

    num_dimensions = 3

    for file_num, file in enumerate(files):

        h5file = h5py.File(file, mode='r')

        features = []

        for window_name in h5file['gabor_features']:
            means = h5file['gabor_features'][window_name]['means']
            stds = h5file['gabor_features'][window_name]['stds']

            concat = X = np.hstack((means, stds))

            features.append(concat)

        full_features = np.array([np.array(f).flatten() for f in zip(*features)])

        full_features = full_features[:1400]

        full_features = full_features.astype(np.float64)

        mds = MDS(num_dimensions, max_iter=100, n_init=1)

        #tsne = TSNE(n_components=num_dimensions)

        Y = mds.fit_transform(full_features)

        #Y = tsne.fit_transform(full_features)

        Ys.append(Y)

    # %% plot 3d
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = sns.color_palette("cubehelix", len(Ys))

    fig = plt.figure()

    if num_dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    labels = ["Raffia", "Brick"]

    data = []
    for i, dataset in enumerate(Ys):
        if num_dimensions == 3:
            ax.scatter(dataset[:, 0],
                       dataset[:, 1],
                       dataset[:, 2],
                       c=colors[i],
                       marker="+",
                       label=os.path.basename(labels[i]))

            trace = go.Scatter3d(
                x=dataset[:, 0],
                y=dataset[:, 1],
                z=dataset[:, 2],
                mode='markers',
                name=labels[i],
                marker=dict(
                    size=12,
                    symbol='cross',
                    line=dict(
                        width=0
                    ),
                    opacity=0.9
                )
            )

            data.append(trace)
        else:
            ax.scatter(dataset[:, 0],
                       dataset[:, 1],
                       c=colors[i],
                       marker="+",
                       label=os.path.basename(labels[i]))

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')
    #plt.legend()

    #plt.show()