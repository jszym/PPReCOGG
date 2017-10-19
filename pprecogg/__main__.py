# -*- coding: utf-8 -*-
#
# __main__.py
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

import click
import numpy as np

@click.group()
def cli_pprecogg():
    pass


@cli_pprecogg.command()
@click.option('--config_file', help='Path to a JSON file that contains the paths to\
                                        Gabor features.')
def visualise_gabor_features(config_file):
    from . import visualiseFeatures

    config = parse_full_auto_config(config_file)
    visualiseFeatures.visualise(config)


@cli_pprecogg.command()
@click.option('--filename', prompt='File path to extract features from:',
              help='File to extract features from:')
@click.option('--resize', default=255, help='Size of image to be resized\
                                                for internal calculations')
@click.option('--win_sizes', default=[64,32,
                                      16,8,4], help='Size of windows to extract.')
@click.option('--batch_size', default=100, help='Size of batches for internal\
                                                    calculations')
@click.option('--gabor_sigma', default=1.0, help='Gabor σ parameter for\
                                                   internal calculations')
@click.option('--gabor_lambda', default=0.25, help='Gabor λ parameter for\
                                                    internal calculations')
@click.option('--gabor_gamma', default=0.02, help='Gabor γ parameter for\
                                                   internal calculations')
@click.option('--gabor_psi', default=0, help='Gabor ψ parameter for\
                                                   internal calculations')
@click.option('--gabor_thetas', default=[np.pi/4,
                                         np.pi/2,
                                         3*np.pi/4,
                                         np.pi], help='Gabor θ parameters for\
                                                   internal calculations')
def extract_gabor_features(filename, resize, win_sizes,
                            batch_size, gabor_sigma, gabor_lambda,
                            gabor_gamma, gabor_psi, gabor_thetas):
    """Tool for extracting Gabor features"""
    import gaborExtract

    gaborExtract.extract_gabor_features(filename, resize, win_sizes,
                            batch_size, gabor_sigma, gabor_lambda,
                            gabor_gamma, gabor_psi, gabor_thetas)


@cli_pprecogg.command()
@click.option('--config_file', help='Path to a JSON file that contains the paths to\
                                        Gabor features.')
def classify_pixel_features(config_file):
    """
    Classifies pixel features as belonging to one of a known set.
    """

    import classifyFeatures, json

    with open(config_file) as f:
        config = json.load(f)

    if not ("known_features" in config and "unknown_feature" in config):
        raise Exception("Config file missing either known_features or unknown_features")

    if not isinstance(config["unknown_feature"], str):
        raise Exception("unknown_features must be strings")

    if not isinstance(config["known_features"], list):
        raise Exception("known_features must be list")

    for item in config["known_features"]:
        if not isinstance(item, str):
            raise Exception("known_features elements must be strings")

    unknown_features_path = config["unknown_feature"]
    known_features_paths = config["known_features"]

    classifyFeatures.classify_features(unknown_features_path,
                                       known_features_paths,
                                       True)


def parse_full_auto_config(config_file):
    import json

    with open(config_file, 'r') as f:
        config = json.load(f)

    expected_arguments = [
        ("unknown_image", None),
        ("known_images", False),
        ("unknown_features", False),
        ("known_features", False),
        ("resize", 255),
        ("win_sizes", [64, 32, 16, 8, 4]),
        ("batch_size", 100),
        ("gabor_sigma", 1.0),
        ("gabor_lambda", 0.25),
        ("gabor_gamma", 0.02),
        ("gabor_psi", 0),
        ("gabor_thetas", [np.pi/4, np.pi/2, 3*np.pi/4, np.pi]),
        ("gabor_ksize", "win_shape")
    ]

    for argument in expected_arguments:
        if argument[0] not in config:
            if argument[1] is None:
                raise Exception("Expected {} when parsing config, but none found".format(argument[0]))

            config[argument[0]] = argument[1]

    return config

@cli_pprecogg.command()
@click.option('--config_file', help='Path to a JSON file that contains the paths to\
                                        images to train and classify on.')
def full_auto(config_file):
    """
    Train, classify and visualise images.
    """

    import pickle
    import gaborExtract, classifyFeatures

    config = parse_full_auto_config(config_file)

    if config["unknown_features"] is not False:

        unknown_features_path = config["unknown_features"]

    else:
        if config["unknown_image"] is False:
            raise Exception("Either unknown_features or unknown_images must be specified")
        unknown_features_path = gaborExtract.extract_gabor_features(config["unknown_image"],
                                                                config["resize"],
                                                                config["win_sizes"],
                                                                config["batch_size"],
                                                                config["gabor_sigma"],
                                                                config["gabor_lambda"],
                                                                config["gabor_gamma"],
                                                                config["gabor_psi"],
                                                                config["gabor_thetas"],
                                                                config["gabor_ksize"])

    if config["known_features"] is not False:
        known_features_paths = config["known_features"]
    else:
        if config["known_images"] is False:
            raise Exception("Either known_features or known_images must be specified")

        known_features_paths = []

        for known_image_path in config["known_images"]:
            features = gaborExtract.extract_gabor_features(known_image_path,
                                                           config["resize"],
                                                           config["win_sizes"],
                                                           config["batch_size"],
                                                           config["gabor_sigma"],
                                                           config["gabor_lambda"],
                                                           config["gabor_gamma"],
                                                           config["gabor_psi"],
                                                           config["gabor_thetas"],
                                                           config["gabor_ksize"])

    class_names, classified_coords = classifyFeatures.classify_features(unknown_features_path,
                                                                        known_features_paths)

    classified_coords_dict = {class_names[class_num]: class_coords for class_num, class_coords in enumerate(classified_coords)}

    with open("output.pickle", "wb") as f:
        pickle.dump(classified_coords_dict, f)

    classifyFeatures.plot_coords(classified_coords_dict,
                                 config["unknown_image"],
                                 config["resize"])

cli = click.CommandCollection(sources=[cli_pprecogg])

if __name__ == '__main__':
    cli()

