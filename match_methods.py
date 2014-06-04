
"""
A few methods to compare with grid match

Jonas Toft Arnfred, 2013-11-18
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
from itertools import chain
from matchutil import get_features, bf_match, flann_match
import fastmatch
import fastmatch_alt

####################################
#                                  #
#           Functions              #
#                                  #
####################################

def grid_v1(query_cache, im_target, options = {}) :
    default_options = {"evaluate_verbose" : True,
                "thumb_strategy" : lambda n : 1.1,
                "neighbor_strategy" : lambda n : 0.9,
                "grid_size" : (90, 90),
                "grid_margin" : 25,
                "radius" : 50,
                "dist_threshold" : 50,
                "depth_first" : True,
                "thumb_size" : (300, 300),
                "log" : None}
    for k, v in default_options.items() :
        options[k] = v
    return fastmatch.match(query_cache, im_target, options = options)


def grid_v2(query_data, im_target, options = {}) :
    default_options = {"evaluate_verbose" : True,
                "thumb_strategy" : lambda n : 1.1,
                "neighbor_strategy" : lambda n : 0.9,
                "grid_size" : (90, 90),
                "grid_margin" : 25,
                "radius" : 50,
                "dist_threshold" : 50,
                "depth_first" : True,
                "thumb_size" : (300, 300),
                "log" : None}
    for k, v in default_options.items() :
        options[k] = v
    return fastmatch_alt.match(query_data, im_target, options = options)


def bf(query_cache, im_target, options = {}) :
    t_keypoints, t_descriptors = get_features(im_target)
    q_descriptors = query_cache.original["descriptors"]

    matches = [m for m in bf_match(q_descriptors, t_descriptors, options = options) if len(m) > 0]
    ratios = numpy.array([m[0].distance / m[1].distance for m in matches])

    q_pos = query_cache.original["positions"]
    t_pos = numpy.array([t_keypoints[m[0].trainIdx].pt for m in matches])

    positions = numpy.array(zip(q_pos, t_pos))

    def get_matches(tau) :
        return [{ "positions" : (p_q, p_t), "ratio" : r }
                for (p_q, p_t), r in zip(positions, ratios) if r < tau]

    return get_matches



def flann(query_cache, im_target, options = {}) :
    # Doesn't work for k > 2
    t_keypoints, t_descriptors = get_features(im_target)
    q_descriptors = numpy.array(query_cache.original["descriptors"], dtype=numpy.float32)

    matches = flann_match(q_descriptors, t_descriptors, k=2, options = options)
    ratios = numpy.array([m[0].distance / m[1].distance for m in matches])

    q_pos = query_cache.original["positions"]
    t_pos = numpy.array([t_keypoints[m[0].trainIdx].pt for m in matches])

    positions = numpy.array(zip(q_pos, t_pos))

    def get_matches(tau) :
        return [{ "positions" : (p_q, p_t), "ratio" : r }
                for (p_q, p_t), r in zip(positions, ratios) if r < tau]

    return get_matches


def mirror(query_cache, im_target, options = {}) :
    t_keypoints, t_descriptors = get_features(im_target)
    q_descriptors = numpy.array(query_cache.original["descriptors"], dtype=numpy.float32)
    q_pos = query_cache.original["positions"]

    all_descriptors = numpy.concatenate((numpy.array(q_descriptors, dtype=numpy.uint8), numpy.array(t_descriptors, dtype=numpy.uint8)))
    matches = [m for m in bf_match(all_descriptors, all_descriptors, k=3) if len(m) > 0]
    ratios = numpy.array([m[1].distance / m[2].distance if len(m) > 0 else 999 for m in matches])
    positions = list(chain(q_pos, (kp.pt for kp in t_keypoints)))

    def get_matches(tau) :
        def iter() :
            already_matched = {}
            for r, m in zip(ratios, matches) :
                # Check that matches are in separate images
                if (m[1].trainIdx >= len(q_pos)) == (m[1].queryIdx < len(q_pos)) :
                    # Check that match hasn't been matched already
                    q_index = min(m[1].trainIdx, m[1].queryIdx)
                    t_index = max(m[1].trainIdx, m[1].queryIdx)
                    if not already_matched.get((q_index, t_index), False) :
                        already_matched[(q_index, t_index)] = True
                        yield { "positions" : (q_pos[q_index], positions[t_index]), "ratio" : r }
        return list(iter())

    return get_matches
