"""
Utilities for matching features from opencv in easy to use wrapper functions

Jonas Toft Arnfred, 2013-05-05
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2


#########################################
#                                       #
#               Matching                #
#                                       #
#########################################

def get_features(data, feature_type = "SIFT") :
    # find the keypoints and descriptors with SIFT
    if feature_type == "SIFT" :
        return cv2.SIFT().detectAndCompute(data, None)

def get_keypoints(data, feature_type = "SIFT") :
    if feature_type == "SIFT" :
        return cv2.SIFT().detect(data)


def bf_match(dt1, dt2, k = 1, options = {}) :
    """ Use opencv's matcher to bruteforce nearest neighbors """
    crossCheck = k == 1 and options.get("crossCheck", False) == True
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck)
    return matcher.knnMatch(dt1, dt2, k = k)


def flann_match(dt1, dt2, k = 1, options = {}) :
    """ Match two sets of descriptors """
    algorithm = options.get("algorithm", 1)
    trees = options.get("trees", 5)
    checks = options.get("checks", 200)
    # Construct approximate nearest neighbor tree
    # "algorithm"     : {
    #    "linear"    : 0,
    #    "kdtree"    : 1,
    #    "kmeans"    : 2,
    #    "composite" : 3,
    #    "kdtree_simple" : 4,
    #    "saved": 254,
    #    "autotuned" : 255,
    #    "default"   : 1
    #},
    index_params = dict(algorithm = algorithm, trees = trees)
    search_params = dict(checks = checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match features
    return flann.knnMatch(dt1, dt2, k = k)
