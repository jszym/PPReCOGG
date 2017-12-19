# -*- coding: utf-8 -*-
#
# classifyFeatures.py
# ========================
# Classify extracted Gabor features according
# to a set of known Gabor features.
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

import h5py
import csv
import operator
import numpy as np
import os.path
from sklearn.neighbors import KNeighborsClassifier


def parse_features_hdf5(file_path):
    """
    Reads Gabor features from HDF5 files. Outputs a tuple of coordinates and
    features. The coordinates are those of which feature windows have been
    centered on. Features are an array of feature means and standard deviations
    of Gabor energies.

    :param file_path: Path to the HDF5 file with the Gabor features.
    """
    h5file = h5py.File(file_path, mode='r')

    features = []

    for window_name in h5file['gabor_features']:
        features.append(h5file['gabor_features'][window_name]['means'])
        features.append(h5file['gabor_features'][window_name]['stds'])

    coord = h5file['gabor_features'][window_name]['coord']

    return coord, features


def flatten_features(file_paths):
    """
    Given a number of Gabor features stored in HDF5 files, create
    three arrays containing the coordinates, features, and labels
    across all feature files.

    :param file_paths: an array with the paths to the feature files.
    """

    flat_feature_coords = []
    flat_feature_values = []
    flat_feature_labels = []

    num_known_features = len(file_paths)

    print("Processing {} files...".format(num_known_features))

    for file_num, file in enumerate(file_paths):

        print("[{}/{}] Extracting features from {}".format(file_num + 1,
                                                           num_known_features,
                                                           file))

        coord, features = parse_features_hdf5(file)

        if len(flat_feature_coords) == 0:
            flat_feature_coords = coord
        else:
            flat_feature_coords = np.vstack((flat_feature_coords,
                                                coord))

        flat_features = np.array([np.array(f).flatten() for f in zip(*features)])
        labels = [file] * len(flat_features)

        if file_num == 0:
            flat_feature_values = flat_features
            flat_feature_labels = labels

        else:
            flat_feature_values = np.vstack((flat_feature_values,
                                              flat_features))

            flat_feature_labels = np.hstack((flat_feature_labels,
                                              labels))

    return flat_feature_values, \
            flat_feature_coords, \
            flat_feature_labels


def classify_features(unknown_features_path, known_features_paths, save_csv=False):
    """
    Classifies features, given multiple paths to features of known classes, and one
    path to features of an unknown image that is comprised of one or more of the 
    known classes. Outputs two arrays. The first contains the labels of the pixels
    classified, and the second contains the coordinates of the pixels classified.
    
    :param unknown_features_path: Path to the HDF5 files that contain the Gabor 
                                    features of the image which is to be 
                                    classified.
    :param known_features_paths: Array of paths to the HDF5 files containing the
                                    features of known classes.
    :param save_csv: Boolean. Whether or not to save the list of pixels with their
                        labels in a comma-seperated-value file (CSV). Can be opened
                        easily in Excel and parsed by other libraries.
    """

    known_feature_values, \
    known_feature_coords, \
    known_feature_labels = flatten_features(known_features_paths)

    # notice here, we throw away the labels because they're not
    # useful for the unknown features
    unknown_feature_values, \
    unknown_feature_coords, \
    _, = flatten_features([unknown_features_path])

    feature_knn = KNeighborsClassifier(n_neighbors=3)
    feature_knn.fit(known_feature_values,
                  known_feature_labels)

    unknown_knn_predictions = feature_knn.predict(unknown_feature_values)

    class_labels = {name: num for num, name in enumerate(list(set(unknown_knn_predictions)))}

    coded_predictions = np.array([class_labels[i] for i in unknown_knn_predictions])

    classified_coords = []

    class_names = []
    class_names_row = []
    coord_axis_row = []

    for class_code in range(len(set(unknown_knn_predictions))):
        classified_coords.append(unknown_feature_coords[coded_predictions == class_code])

    for class_name in sorted(class_labels.items(), key=operator.itemgetter(1)):
        class_names.append(class_name[0])
        class_names_row.append(class_name[0])
        class_names_row.append('')
        coord_axis_row.append('x')
        coord_axis_row.append('y')

    if save_csv:

        print("Writing CSV file...")

        with open('output.csv', 'w', newline='') as csv_file:

            output_csv = csv.writer(csv_file, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            output_csv.writerow(class_names_row)
            output_csv.writerow(coord_axis_row)

            for coords in zip(*classified_coords):
                coord_row = []
                for coord in coords:
                    for axis in coord:
                        coord_row.append(axis)
                output_csv.writerow(coord_row)

        print("Done.")

    return class_names, classified_coords


def plot_coords(class_coords, unknown_image, resize):
    import matplotlib.pyplot as plt
    import skimage.transform

    im = plt.imread(unknown_image)

    if resize:
        if resize > 1:
            im = skimage.transform.resize(im,
                                          (resize,
                                           resize))
        else:
            im = skimage.transform.resize(im,
                                          (im.shape[0] * resize,
                                           im.shape[1] * resize))

    plt.imshow(im, cmap="Greys_r")

    for class_name in class_coords:
        coords = class_coords[class_name]
        plt.scatter(x=coords[:, 0], y=coords[:, 1], s=4, alpha=0.2, label=os.path.basename(class_name))

    #plt.legend()
    plt.show()