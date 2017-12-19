Examples
========

Library Usage Example
---------------------
You can use PPReCOGG from python as so::

    from pprecogg import gaborExtract, classifyFeatures

    # path to the image you wish to classify
    unknown_img_path = "/path/to/unknown/image"

    # paths to the images whose class you know
    adh_img_path = "/path/to/adh/image"
    dcis_img_path = "/path/to/dcis/image"

    # features are extracted into HDF5 files, and extract_gabor_features
    # returns the path to said file
    unknown_features_path = gaborExtract.extract_gabor_features(unknown_img_path)
    adh_features_path = gaborExtract.extract_gabor_features(adh_img_path)
    dcis_features_path = gaborExtract.extract_gabor_features(dcis_img_path)

    # classify features from unknown image. 
    # returns an array of class ID and an array of classified coordinates
    # indexed by class (see: that array of class IDs)
    class_names,classified_coords = classifyFeatures.classify_features(unknown_features_path,
    known_features_paths)


    # we can convert this into a dictionary where the key is the class name
    # and the value are the coordinates that belong to it
    classified_coords_dict = {class_names[class_num]: class_coords for class_num, class_coords in enumerate(classified_coords)}

    # small ergonomic function to plot classified pixels on to the
    # unknown image
    classifyFeatures.plot_coords(classified_coords_dict,
                                    unknown_img_path)



CLI Usage
---------

Simplest way to use PPReCOGG in CLI mode is to use the `full_auto`
mode.

Step One: Create configuration file::

    {
    "unknown_image": "/path/to/unknown/image",

    /* optional, for rerunning */
    "unknown_features": "/path/to/unknown/features.h5",

    "known_images":[
        "/path/to/known/image",
        "/path/to/known/image"],

    /* optional, for rerunning */
    "known_features":[
        "/path/to/known/features.h5",
        "/path/to/known/features.h5"
    ],

    /* 
        the smaller, the faster the computations. 
        the bigger, the higher resolution output.
    */
    "resize": 510
    }


Step Two: Run PPReCOGG in ``full_auto`` mode

``python -m pprecogg full_auto --config_file config.json``