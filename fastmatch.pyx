# encoding: utf-8
# cython: profile=True
# filename: fastmatch.pyx
"""
Fast matching algorithm for image matching

Jonas Toft Arnfred, 2013-05-05
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

from matchutil import get_features
from cache import Grid_Cache, Metric_Cache
import cv2
from imaging import get_thumbnail, get_size
import numpy
import itertools
cimport numpy
from cache cimport Metric_Cache, Grid_Cache


####################################
#                                  #
#           Functions              #
#                                  #
####################################

def match(Metric_Cache query_cache, numpy.ndarray[numpy.uint8_t, ndim=3] target_img, object options = {}) :
    # Declare datatypes
    cdef int thumb_x, thumb_y, grid_x, grid_y, grid_margin, radius
    cdef double thumb_tau
    cdef numpy.ndarray thumb_positions, thumb_ratios, positions_iter
    # Get parameters
    thumb_x, thumb_y = options.get("thumb_size", (400, 400))
    grid_x, grid_y = options.get("grid_size", (50, 50))
    thumb_strategy = options.get("thumb_strategy", lambda n : n)
    log = options.get("log", None)
    grid_margin = options.get("grid_margin", 25)
    radius = options.get("radius", 100)
    # Create target cache
    cdef Grid_Cache target_cache = Grid_Cache(target_img, (grid_x, grid_y), get_features, margin = grid_margin)
    thumb_positions, thumb_ratios = match_thumbs(target_img, query_cache, thumb_x = thumb_x, thumb_y = thumb_y)
    # Create a function where we can wary tau to get different results
    def get_matches(double tau) :
        thumb_tau = thumb_strategy(tau)
        positions_iter = itertools.chain(thumb_positions[thumb_ratios<thumb_tau])
        matches = do_iter(positions_iter, query_cache, target_cache, tau = tau, thumb_tau = thumb_tau, radius = radius, log = log)
        return matches
    return get_matches


cdef object do_iter(object positions, Metric_Cache cache, Grid_Cache target_grid, double tau, double thumb_tau, int radius, object log = None) :
    # Declare datatypes
    cdef numpy.ndarray[numpy.double_t] query_pos, target_pos
    cdef numpy.ndarray[numpy.double_t, ndim=2] p
    cdef numpy.ndarray result_pos, ratios
    cdef int col, row, query_col, query_row
    cdef double r
    matches = []
    has_matched = {}
    found_matches = {}
    while True :
        try :
            query_pos, target_pos = positions.next()
            col, row = target_grid.block(target_pos[0], target_pos[1])
            query_col, query_row = target_grid.block(query_pos[0], query_pos[1])
            if not has_matched.get((col, row, query_col, query_row), False) :
                has_matched[(col, row, query_col, query_row)] = True
                result_pos, ratios, query_idx = match_position((query_pos, target_pos), cache, target_grid, radius = radius)
                # For each match we don't discard, we might want to examine the neighbor field
                neighbors = get_neighbors(target_pos, result_pos[ratios<tau], target_grid)
                if len(neighbors) > 0 :
                    positions = itertools.chain(neighbors, positions)
                # Log if we have to
                if log != None :
                    log.append(log_round(query_pos, target_pos, result_pos, target_grid, ratios, tau, radius))
                # Yield result if it hasn't been yielded already
                for p, r, index in zip(result_pos[ratios<tau], ratios[ratios<tau], query_idx[ratios<tau]) :
                    p_touple = map(int, (p[0,0], p[0,1], p[1,0], p[1,1]))
                    if p_touple not in found_matches.get(r,[]) :
                        found_matches[r] = found_matches.get(r,[]) + [p_touple]
                        matches.append((index, { "positions" : p, "ratio" : r }))
        except StopIteration :
            break
    return dict(matches)


cdef object get_neighbors(numpy.ndarray[numpy.double_t] target_pos,
                         numpy.ndarray result_pos,
                         Grid_Cache target_grid) :
    cdef int col, row
    cdef numpy.ndarray[numpy.int_t] neighbor_pos
    col, row = target_grid.block(target_pos[0], target_pos[1])
    neighbors = []
    for p_query, p_target in result_pos :
        neighbor_pos = target_grid.get_neighbor(col, row, p_target[0], p_target[1])
        if neighbor_pos[0] != -1 :
            neighbors.append(numpy.array((p_query, neighbor_pos)))
    return neighbors



cdef object match_thumbs(numpy.ndarray[numpy.uint8_t, ndim=3] img, Metric_Cache cache, int thumb_x = 400, int thumb_y = 400) :
    cdef int t_orig_x, t_orig_y
    cdef double t_ratio_x, t_ratio_y, q_ratio_x, q_ratio_y
    cdef numpy.ndarray t_pos, q_pos, q_distances
    cdef numpy.ndarray ratios, q_ratio, indices, pos_scaled
    # Load target and find descriptors and size
    cdef numpy.ndarray[numpy.uint8_t, ndim=3] target = get_thumbnail(img, (thumb_x, thumb_y))
    t_orig_x, t_orig_y = get_size(img)
    t_keypoints, t_descriptors = get_features(target)

    # Similar for query
    q_descriptors = cache.thumb["descriptors"]
    q_distances = cache.thumb["distances"]
    q_pos = cache.thumb["positions"]
    # match thumbnails and find ratio
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = [m for m in matcher.knnMatch(q_descriptors, t_descriptors, k=1) if len(m) > 0]
    ratios = numpy.array([m[0].distance / q_distances[m[0].queryIdx] for m in matches])

    # Get positions of points and their scaling factor
    t_pos = numpy.array([t_keypoints[m[0].trainIdx].pt for m in matches])
    t_ratio_x = t_orig_x/float(target.shape[1])
    t_ratio_y = t_orig_y/float(target.shape[0])
    t_ratio = numpy.array([t_ratio_x, t_ratio_y])

    q_pos = numpy.array([cache.thumb["positions"][m[0].queryIdx] for m in matches])
    q_ratio_x = cache.original["size"][0]/float(cache.thumb["size"][0])
    q_ratio_y = cache.original["size"][1]/float(cache.thumb["size"][1])
    q_ratio = numpy.array([q_ratio_x, q_ratio_y])

    # Sort ratios and scale positions
    indices = numpy.argsort(ratios)
    pos_scaled = numpy.array([(q_p * q_ratio, t_p * t_ratio) for q_p, t_p in zip(q_pos, t_pos)])

    return pos_scaled[indices], ratios[indices]


# Match point strategy #1:
cdef match_position(pos, Metric_Cache cache, Grid_Cache target, int radius = 100) :
    cdef numpy.ndarray cache_pos, cache_dis, ratios, positions, ratio_indices
    cdef int r, offset_x, offset_y, target_x, target_y, query_x, query_y
    # Find positions
    query_x, query_y = pos[0]
    target_x, target_y = pos[1]

    cache_ds, cache_pos, cache_dis, cache_idx = cache.get(query_x, query_y, radius)

    target_kp, target_ds = target.get(target_x, target_y)
    if target_ds == None :
        return numpy.array([]), numpy.array([]), numpy.array([])
    offset_x, offset_y = target.offset(target_x, target_y)
    target_pos = [numpy.array([k.pt[0]+offset_x, k.pt[1]+offset_y]) for k in target_kp]

    # Match descriptors using bf
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = [m for m in matcher.knnMatch(cache_ds, target_ds, k=1) if len(m) > 0]

    # Distances to nearest neighbor and positions
    ratios = numpy.array([m[0].distance / cache_dis[m[0].queryIdx] for m in matches])
    positions = numpy.array([(cache_pos[m[0].queryIdx], target_pos[m[0].trainIdx]) for m in matches])

    return positions, ratios, cache_idx


cdef log_round(numpy.ndarray query_pos, numpy.ndarray target_pos, numpy.ndarray result_pos, Grid_Cache target_grid, numpy.ndarray ratios, double tau, int radius) :
    return {
        "query_pos" : query_pos,
        "target_pos" : target_pos,
        "target_grid" : target_grid.last,
        "matches" : result_pos[ratios<tau],
        "radius" : radius,
        "ratios" : ratios[ratios<tau],
        "margin" : target_grid.margin }
