# -*- coding: utf-8 -*-
#
# gaborExtract.py
# ========================
# Library for extracting gabor features from windows of an image.
# Accelerated by GPU via the Theano library
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

from PIL import Image
import numpy as np
import skimage.transform
import theano.tensor as T
import theano
import math
import cv2
import time
import tables
import os.path
import random
import itertools


def get_window(i, j, win_size, im):
    """
    Get a window from `im` of size `win_size` centered
    on coordinates `(i,j)`.

    :param i:  x pixel coordinate to center the window on the image.
    :type i: int

    :param j: y pixel coordinate to center the window on the image.
    :type j: int

    :param win_size: the length and width of the window to extract.
    :type win_size: int

    :param im: numpy pixel array of the image.
    :type im: numpy array

    :return: the window extracted of size (win_size,win_size).
    :rtype: numpy array (2 dimensions)
    """

    bottom_row = j - int(0.5 * win_size)
    top_row = j + int(0.5 * win_size)

    left_column = i - int(0.5 * win_size)
    right_column = i + int(0.5 * win_size)

    return im[bottom_row:top_row, left_column:right_column]


def get_windows_batches(num_windows, win_sizes, im, batch_size):
    """
    Get `num_windows` number of randomly places windows of sizes
    `win_sizes` from image `im` in batches of `batch_size`.

    This is a generator, and will have to be iterated over.

    :param num_windows: Number of windows (of each window size)
                        to extract from image.
    :type num_windows: int

    :param win_sizes: Size of windows to be extracted. Final
                        number of windows extracted will be
                        `window_size * num_windows`.
    :type win_sizes: array of ints

    :param im: Array of pixels of the image from which to extract
                windows from.
    :type im: numpy array

    :param batch_size: Max number of windows per batch (iteration).
    :type batch_size: int

    :return: Batch of randomly sampled windows from `im`. 3 dimensions
                (batch_size, window_size, windows_size).
    :rtype: numpy array
    """

    max_win_size = np.max(win_sizes)

    # randomly sample the entire population of coords
    population_coords = itertools.product(range(max_win_size // 2,
                                                im.shape[0] - (max_win_size // 2)),
                                          repeat=2)

    population_coords = [_ for _ in population_coords]

    coords = random.sample(population_coords, num_windows)

    yield coords

    for win_size in win_sizes:
        batch = []
        batch_coords = []
        i = 0
        for coord in coords:
            window = get_window(coord[0],
                                coord[1],
                                win_size,
                                im)

            batch.append([window])
            batch_coords.append(coord)

            i += 1
            if i % batch_size == 0:
                yield (batch_coords, np.array(batch))
                batch = []
                batch_coords = []

        if len(batch) > 0:
            yield (batch_coords, np.array(batch))


def get_HDF5_table(hd5_filename, window_sizes, num_thetas):
    """
    Opens HDF5 file and tables to save gabor means and standard
    deviations

    :param hd5_filename: Name of the file to save the features to.
    :type hd5_filename: String

    :param window_sizes: A list of window sizes whose means and stdevs
                            will be saved.
    :type window_sizes: Array

    :param num_thetas: Number of thetas (i.e number of means and stdevs)
    :type num_thetas: Int

    :return: List of pytable objects
    """
    h5file = tables.open_file(hd5_filename,
                              mode="w")

    group_id = "gabor_features"
    group_title = "Gabor Features"
    h5_group = h5file.create_group("/",
                                   group_id,
                                   group_title)

    class GaborWinFeatures(tables.IsDescription):
        win_size = tables.Int32Col(pos=0)
        coord = tables.Int32Col(shape=(2,), pos=1)
        means = tables.Float32Col(shape=(num_thetas,), pos=2)
        stds = tables.Float32Col(shape=(num_thetas,), pos=3)

    gabor_feature_tables = {}

    for window_size in window_sizes:
        table = h5file.create_table(h5_group,
                                    'gabor_features_{}'.format(window_size),
                                    GaborWinFeatures,
                                    "Gabor Features {}".format(window_size))

        gabor_feature_tables[window_size] = table

    return gabor_feature_tables


def extract_gabor_features(filename, resize=255, window_sizes=[64, 32, 16, 8, 4],
                           batch_size=100, gabor_sigma=1.0, gabor_lambda=0.25,
                           gabor_gamma=0.02, gabor_psi=0,
                           gabor_thetas=[np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                           gabor_ksize="win_size"):
    """
    Briefly, calculates the means and standard deviations of a random sampling of
    windows convolved with several Gabor filters.

    Windows of sizes `window_sizes` are centered on a random sampling of pixels of
    the image located at `filename` resized to `resize`x`resize` pixels. A quarter
    of the area of the image is sampled.

    The final output are the means and standard deviation of the
    resultant convolutions of the windows against `n` Gabor filters, where
    `n` is the number of elements in `gabor_thetas`. The Gabor filters are
    are generated according to the `gabor_` prefixed arguments.

    The means and standard deviation are saved into a HDF5 file with the filename
    "gabor_{image_name}_s{sigma}_l{lambda}_g{gamma}_p{psi}_{time}.h5" where

    {image_name} is the filename of the image
    {sigma} is the sigma argument of the gabor function
    {lambda} is the lambda argument of the gabor function
    {gamma} is the gamma argument of the gabor function
    {psi} is the psi argument of the gabor function
    {time} is the unix time stamp

    If a compliant GPU is detected, convolutions will be computed on them.
    Computations will fall back on CPU in other cases.

    Convolutions are computed in batches of size `batch_size`. In the event of a
    memory error, this number should be reduced.

    :param filename: The path to the image whose gabor energies are to be calculated.
    :param resize: Image at `filename` will be resized to `resize`x`resize` pixels
                    prior to computation. Image size is positively correlated to
                    computation time.
    :param window_sizes: The sizes of the sliding windows to be extracted at each
                            pixel.
    :param batch_size:  The number of windows to convolve at a time. Reduce this
                            number in the event of memory errors; increase this
                            number in the event of slow computation time.
    :param gabor_sigma: The standard-deviation of the gabor function.
    :param gabor_lambda: The frequency of the gabor function.
    :param gabor_gamma: The bandwidth of the gabor function.
    :param gabor_psi: The phase offset of the gabor function.
    :param gabor_thetas: The orientation of the filter, in radians.
    :param gabor_ksize: Defaults to (19,19). When gabor_ksize is equal to the
                        string "win_size", the gabor kernel is the size of the
                        window it's convolving.
    :return: The name of the H5 file that the results are saed to.
    """
    image_name = os.path.basename(filename)

    hd5_filename = "gabor_{}_s{}_l{}_g{}_p{}_{}.h5".format(image_name,
                                                           gabor_sigma,
                                                           gabor_lambda,
                                                           gabor_gamma,
                                                           gabor_psi,
                                                           int(time.time()))

    gabor_feature_tables = get_HDF5_table(hd5_filename,
                                          window_sizes,
                                          len(gabor_thetas))

    im = Image.open(filename).convert('L')

    im = np.array(im)

    if resize:
        if resize > 1:
            im = skimage.transform.resize(im,
                                          (resize,
                                           resize))
        else:
            im = skimage.transform.resize(im,
                                          (im.shape[0] * resize,
                                           im.shape[1] * resize))

    # the number of windows (of each size) that are sampled from the image
    # TODO: Make this an argument, it' currently set at the magic number of 1/4
    num_windows = int(math.pow(im.shape[0], 2) // 4)

    batches = get_windows_batches(num_windows,
                                  window_sizes,
                                  im, batch_size)

    total_batches = round(len(next(batches))/100)
    total_time = 0
    batch_counter = 0

    image_tensor = T.tensor4()
    filter_tensor = T.tensor4()

    for i, batch in enumerate(batches):
        start = time.time()

        coord, batch = batch

        win_shape = batch[0, 0, :, :].shape
        num_batches = len(batch)

        batch = batch.astype(np.float32)

        kerns = []

        for gabor_theta in gabor_thetas:

            if gabor_ksize == "win_shape":
                gabor_ksize = win_shape

            kern = cv2.getGaborKernel(gabor_ksize,
                                      gabor_sigma,
                                      gabor_theta,
                                      gabor_lambda,
                                      gabor_gamma,
                                      gabor_psi, ktype=cv2.CV_32F)
            kerns.append(kern)

        means = []
        stds = []
        for kern in kerns:
            kern_shape = np.array(kern).shape

            kern_4d = np.array(kern).reshape(1,
                                             1,
                                             kern_shape[0],
                                             kern_shape[1]).astype(np.float32)

            theano_convolve2d = theano.function([image_tensor, filter_tensor],
                                                T.nnet.conv2d(
                                                    image_tensor,
                                                    filter_tensor,
                                                    input_shape=(num_batches, 1, win_shape[0], win_shape[1]),
                                                    filter_shape=(num_batches, 1, kern_shape[0], kern_shape[1]),
                                                    border_mode=(win_shape[0], win_shape[1])))

            theano_convolved = theano_convolve2d(batch,
                                                 kern_4d)

            output_shape = theano_convolved.shape

            conv_flat = theano_convolved.reshape(num_batches,
                                                 output_shape[2] * output_shape[3])

            conv_mean = conv_flat.mean(axis=1)
            means.append(conv_mean)

            conv_std = conv_flat.std(axis=1)
            stds.append(conv_std)

        end = time.time() - start

        total_time += end

        batch_counter += 1

        avg_time = total_time/(batch_counter)

        print("{:5}/{:5} batches complete [ETA {:3.2f}m], ~{:3.2f}s per batch".format(batch_counter,
                                                                                    total_batches,
                                                                                    (avg_time*(total_batches-batch_counter))/60,
                                                                                    avg_time))

        # h5 start
        zipped_means = zip(*means)
        zipped_stds = zip(*stds)

        feature_table = gabor_feature_tables[win_shape[0]]
        feature_table_row = feature_table.row

        for i, mean in enumerate(zipped_means):
            feature_table_row['win_size'] = win_shape[0]
            feature_table_row['coord'] = coord[i]
            feature_table_row['means'] = mean
            feature_table_row['stds'] = next(zipped_stds)
            feature_table_row.append()

        feature_table.flush()
        # h5 end

    return hd5_filename
