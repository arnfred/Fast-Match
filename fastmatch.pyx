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
import cv2
from imaging import get_thumbnail, get_size
import numpy
import itertools
cimport numpy
from cache cimport Metric_Cache, Grid_Cache, Pos
from cache import Grid_Cache, Metric_Cache
from libc.math cimport abs as abs_c

####################################
#                                  #
#           Functions              #
#                                  #
####################################

def match(Metric_Cache query_cache, numpy.ndarray[numpy.uint8_t, ndim=3] target_img, object options = {}) :
    # Declare datatypes
    cdef int thumb_x, thumb_y, grid_x, grid_y, grid_margin, radius
    cdef double thumb_tau, neighbor_tau, radius_scale
    cdef numpy.ndarray thumb_positions, thumb_ratios, positions_iter
    # Get parameters
    thumb_x, thumb_y = options.get("thumb_size", (400, 400))
    grid_x, grid_y = options.get("grid_size", (50, 50))
    thumb_tau = options.get("thumb_tau", 1.0)
    neighbor_tau = options.get("neighbor_tau", 0.9)
    log = options.get("log", None)
    grid_margin = options.get("grid_margin", 25)
    radius = options.get("radius", 100)
    dist_threshold = options.get("dist_threshold", 100)
    depth_first = options.get("depth_first", True)
    # Create target cache
    cdef Grid_Cache target_cache = Grid_Cache(target_img, (grid_x, grid_y), margin = grid_margin)
    thumb_positions, thumb_ratios = match_thumbs(target_img, query_cache, thumb_x = thumb_x, thumb_y = thumb_y)
    # Create a function where we can vary tau to get different results
    def get_matches(double tau) :
        positions = ((p[0], p[1], p[2], p[3]) for p in thumb_positions[thumb_ratios<thumb_tau])
        matches = do_iter(positions, query_cache, target_cache, tau = tau, neighbor_tau = neighbor_tau, radius = radius, dist_threshold = dist_threshold, depth_first = depth_first, log = log)
        return matches
    return get_matches


cdef object do_iter(object positions, Metric_Cache cache, Grid_Cache target_grid, double tau, double neighbor_tau, int radius, int dist_threshold, depth_first = True, object log = None) :
    # Declare datatypes
    cdef numpy.ndarray[numpy.double_t] ratios, pos
    cdef numpy.ndarray[numpy.double_t, ndim=2] result_pos
    cdef Pos p_neighbor
    cdef int round_nb, index, row, col
    matches = []
    has_matched = {}
    found_matches = {}
    round_nb = 1
    while True :
        try :
            current_pos = positions.next()
            if is_new_match(has_matched, current_pos, target_grid, dist_threshold) :
                row, col = target_grid.block(Pos(current_pos[2], current_pos[3]))
                # Match blocks and find all neighbors
                result_pos, ratios = match_position(current_pos, cache, target_grid, radius = radius)
                neighbors = []
                for pos in result_pos[ratios < neighbor_tau] : # pos is in format [query_x, query_y, target_x, target_y]
                    for p_neighbor in target_grid.get_neighbor(row, col, Pos(pos[2],pos[3])) :
                        neighbors.append((pos[0], pos[1], p_neighbor.x, p_neighbor.y))
                positions = itertools.chain(neighbors, positions)
                # Log if we have to
                if log != None :
                    log.append(log_round(current_pos, result_pos, target_grid, ratios, tau, radius))
                    round_nb += 1
                # Yield result if it hasn't been yielded already
                matches.extend(add_matches(result_pos, ratios, tau, found_matches))
        except StopIteration :
            break
    return matches



cdef object add_matches(numpy.ndarray[numpy.double_t, ndim=2] pos,
                        numpy.ndarray[numpy.double_t] ratios,
                        double tau,
                        object found_matches) :
    cdef int i
    cdef double r
    cdef object matches, p_touple, already_matched
    matches = []
    for i in range(ratios.size) :
        r = ratios[i]
        if ratios[i] < tau :
            already_matched = found_matches.get(r,[])
            p_touple = (pos[i,0], pos[i,1], pos[i,2], pos[i,3])
            if len(already_matched) == 0 or p_touple not in already_matched :
                found_matches[r] = already_matched + [p_touple]
                matches.append({ "positions" : ((pos[i,0], pos[i,1]), (pos[i,2], pos[i,3])), "ratio" : ratios[i] })
    return matches


cdef bint is_new_match(object has_matched, object current_pos, Grid_Cache target_grid, int dist_threshold) :
    cdef int round_nb, row, col
    cdef double query_x, query_y
    query_x, query_y = current_pos[0], current_pos[1]
    row, col = target_grid.block(Pos(int(current_pos[2]), int(current_pos[3])))
    # Check if we've matched these blocks before
    last_positions = has_matched.get((row, col), [])
    if len(last_positions) == 0 or above_dist_threshold(last_positions, query_x, query_y, dist_threshold) :
        has_matched[(row, col)] = last_positions + [(query_x, query_y)]
        return True
    return False


cdef bint above_dist_threshold(object last_positions, double cur_x, double cur_y, int dist_threshold) :
    cdef int x, y
    for i in range(len(last_positions)) :
        x = abs_c(last_positions[i][0] - cur_x)
        y = abs_c(last_positions[i][1] - cur_y)
        if x+y < dist_threshold :
            return False
    return True



cdef object match_thumbs(numpy.ndarray[numpy.uint8_t, ndim=3] img, Metric_Cache query_cache, int thumb_x = 400, int thumb_y = 400) :
    cdef int t_orig_x, t_orig_y
    cdef double t_ratio_x, t_ratio_y, q_ratio_x, q_ratio_y
    cdef numpy.ndarray[numpy.int_t, ndim=1] indices
    cdef numpy.ndarray[numpy.int_t, ndim=2] matches
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] q_descriptors, t_descriptors
    cdef numpy.ndarray[numpy.uint8_t, ndim=3] target
    cdef numpy.ndarray[numpy.double_t, ndim=1] ratios, q_distances
    cdef numpy.ndarray[numpy.double_t, ndim=2] positions, t_pos, q_pos, pos_scaled
    # Load target and find descriptors and size
    target = get_thumbnail(img, (thumb_x, thumb_y))
    t_orig_x, t_orig_y = get_size(img)
    t_keypoints, ds = get_features(target)
    t_descriptors = numpy.array(ds, dtype=numpy.uint8)

    # Similar for query
    q_descriptors = numpy.array(query_cache.thumb["descriptors"], dtype=numpy.uint8)
    q_distances = query_cache.thumb["distances"]
    q_pos = query_cache.thumb["positions"]
    t_pos = numpy.array([k.pt for k in t_keypoints])
    # match thumbnails and find ratio
    matches = do_match(q_descriptors, t_descriptors)
    positions = get_positions(matches, q_pos, t_pos)
    ratios = get_ratios(matches, q_distances)

    # Get positions of points and their scaling factor
    q_ratio_x = query_cache.original["size"][0]/float(query_cache.thumb["size"][0])
    q_ratio_y = query_cache.original["size"][1]/float(query_cache.thumb["size"][1])
    t_ratio_x = t_orig_x/float(target.shape[1])
    t_ratio_y = t_orig_y/float(target.shape[0])
    scale_factors = numpy.array([q_ratio_x, q_ratio_y, t_ratio_x, t_ratio_y], dtype=numpy.double)

    pos_scaled = positions * scale_factors
    indices = numpy.argsort(ratios)

    return pos_scaled[indices], ratios[indices]


# Match point strategy #1:
cdef match_position(object pos, Metric_Cache query_cache, Grid_Cache target, int radius = 100) :
    cdef int target_x, target_y, query_x, query_y
    cdef numpy.ndarray[numpy.double_t, ndim=2] positions, target_pos, query_pos
    cdef numpy.ndarray[numpy.double_t, ndim=1] ratios
    cdef numpy.ndarray[numpy.int_t, ndim=1] query_dis
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] target_ds, query_ds
    cdef numpy.ndarray[numpy.int_t, ndim=2] matches
    # Find positions
    query_x, query_y, target_x, target_y = pos

    query_ds, query_pos, query_dis = query_cache.get(query_x, query_y, radius)

    target_pos, target_ds = target.get(Pos(target_x, target_y))
    if (target_ds == None or
        query_ds == None or
        len(target_ds) == 0 or
        len(query_ds) == 0 or
        (len(target_ds) == 1 and
         len(target_ds[0]) == 0)) :
        return numpy.empty((0,0)), numpy.empty(0)

    # Match descriptors using bf
    matches = do_match(query_ds, target_ds)

    # Distances to nearest neighbor and positions
    ratios = get_ratios(matches, numpy.array(query_dis, dtype=numpy.double))
    positions = get_positions(matches, query_pos, target_pos)

    return positions, ratios

cdef numpy.ndarray[numpy.double_t] get_ratios(numpy.ndarray[numpy.int_t, ndim=2] matches, numpy.ndarray[numpy.double_t] query_dis) :
    return matches[0] / query_dis[matches[1]]

cdef numpy.ndarray[numpy.double_t, ndim=2] get_positions(numpy.ndarray[numpy.int_t, ndim=2] matches, numpy.ndarray[numpy.double_t, ndim=2] query_pos, numpy.ndarray[numpy.double_t, ndim=2] target_pos) :
    return numpy.concatenate((query_pos[matches[1]], target_pos[matches[2]]), axis=1)

cdef numpy.ndarray[numpy.int_t, ndim=2] do_match(numpy.ndarray[numpy.uint8_t, ndim=2] query_ds, numpy.ndarray[numpy.uint8_t, ndim=2] target_ds) :
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    return numpy.array([(m[0].distance, m[0].queryIdx, m[0].trainIdx) for m in matcher.knnMatch(query_ds, target_ds, k=1) if len(m) > 0], dtype=numpy.int).T




cdef log_round(object current_pos, numpy.ndarray result_pos, Grid_Cache target_grid, numpy.ndarray ratios, double tau, int radius) :
    return {
        "current_pos" : current_pos,
        "target_cell" : target_grid.last_cell,
        "target_block" : target_grid.last_block,
        "matches" : result_pos[ratios<tau],
        "radius" : radius,
        "ratios" : ratios[ratios<tau],
        "margin" : target_grid.margin }
