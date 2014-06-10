"""
Python module for dealing with the 3D objects dataset published here:
http://www.vision.caltech.edu/pmoreels/Datasets/TurntableObjects/

Jonas Toft Arnfred, 2013-11-18
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
from cache import Metric_Cache
from turntable_ground_truth import get_turntable_path, point_to_index
import imaging
import pandas as pd

####################################
#                                  #
#           Functions              #
#                                  #
####################################


def evaluate(match_fun, angles, object_type, thresholds, ground_truth = None, options = {}) :
    """ Returns number of correct and total matches of match_fun on object:
        match_fun : Function (Function that takes a list of paths and returns matches)
        angles : Int (Rotation in degrees. Must be divisible by 5)
        object_type : String (the object pictured on the turntable)
        thresholds : List[Float] (list of ratio thresholds)
        match_count : List[Int] (List of the amount of possible matches for each feature point)
        options : Dict (Set of parameters for matching etc)
    """

    # Get distance_threshold
    verbose                 = options.get("evaluate_verbose", False)
    print_path              = options.get("print_path", False)

    # Get paths to the three images
    def get_path(i) : return {
            "A" : get_turntable_path(object_type, angles[0] + i*360, "Bottom"),
            "B" : get_turntable_path(object_type, angles[0] + i*360, "Top"),
            "C" : get_turntable_path(object_type, angles[1] + i*360, "Bottom")
        }

    # Get paths
    def get_match_fun(i) :
        # Collect matches
        paths = get_path(i)
        if "baseline_fun" in options :
            options["baseline_cache"] = options["baseline_fun"](object_type, angles[0] + i*360)
            #print("With baseline cache: %s" % options["baseline_cache"].path)
        query_path, target_path = paths["A"], paths["C"]
        query_cache, target_img = Metric_Cache(query_path), imaging.open_img(target_path)
        matches = match_fun(query_cache, target_img, options = options)(thresholds[-1])
        return lambda tau : [m for m in matches if m["ratio"] < tau]

    if print_path :
        paths = get_path(0)
        print("query_path = '%s'" % paths["A"])
        print("target_path = '%s'" % paths["C"])

    # Get matches
    match_funs = map(get_match_fun, range(3))
    matches_df = pd.DataFrame({ img : { tau : f(tau) for tau in thresholds } for img, f in enumerate(match_funs) })

    # function for counting correct and total
    def get_correct(row) :
        gt = ground_truth["correspondences"][row.name]
        def gt_int(p_A) : return [map(int,p) for p, d in gt.get(point_to_index(p_A), [])]
        def pos_int(m) : return map(int, m["positions"][1])
        def count_correct(matches) : return sum([pos_int(m) in gt_int(m["positions"][0]) for m in matches])
        return row.map(count_correct)
    correct = matches_df.apply(get_correct).sum(axis=1)
    def get_total(row) :
        gt = ground_truth["correspondences"][row.name]
        def count_total(matches) :
            return sum([len(gt.get(point_to_index(m["positions"][0]), [])) > 0 for m in matches])
        return row.map(count_total)
    def nb_matches(row) :
        return row.map(len)
    total = matches_df.apply(get_total).sum(axis=1)
    matches = matches_df.apply(nb_matches).sum(axis=1)
    # Get accuracy
    precision = correct / total
    recall = correct / ground_truth["nb_correspondences"]

    if verbose :
        print("""%16s %s: %6i correct of %6i total (%i Matches).\t Precision: %.3f - Recall: %.3f (median)""" % (
                  object_type,
                  angles,
                  sum(correct),
                  sum(total),
                  sum(matches),
                  numpy.median(precision),
                  numpy.median(recall)))

    return { "precision" : precision, "recall" : recall, "correct" : correct, "total" : total }



def evaluate_objects(match_fun, angles, object_types, thresholds, ground_truth_data, options = {}) :
    """ Returns number of correct and total matches of match_fun on objects:
        match_fun : Function (Function that takes a list of paths and returns matches)
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_types : List[String] (the objects pictured on the turntable)
        thresholds : List[Float] (list of ratio thresholds)
        options : Dict (Set of parameters for matching etc)
    """
    # Get correct and total matches for different thresholds at current angle_increment
    def get_results(object_type) :
        gt = ground_truth_data[object_type]
        return evaluate(match_fun, angles, object_type, thresholds, gt, options)


    results = [get_results(o) for o in object_types]
    correct =  { o : r["correct"] for r, o in zip(results, object_types) }
    total =  { o : r["total"] for r, o in zip(results, object_types) }

    return { "correct" : correct, "total" : total }





def get_weights(gt, exclude_keys = []) :
    """ Calculate a set of weights per object and index item (usually angle) """
    nb_corr_accu = { a : sum(map(lambda data : data["nb_correspondences"], gt_item.values()))
                    for a, gt_item in gt.iteritems() if a not in exclude_keys }
    weights = { a : { o : 1.0 / (t["nb_correspondences"] / float(nb_corr_accu[a]))
                     for o, t in gt_item.iteritems() } for a, gt_item in gt.iteritems() if a not in exclude_keys }
    weights_sum = { a : numpy.sum(ow.values()) for a, ow in weights.iteritems() }
    weights_normalize = { a : { o : w / weights_sum[a]
                               for o,w in ow.iteritems() } for a,ow in weights.iteritems() }
    return weights_normalize


def accumulate(results, gt, weights = None, exclude_keys = []) :
    """ Take results from evaluate_objects and ground truth and compile a simple data structure
        containing correct, total and nb_correlations """

    # Get default weights
    if weights == None :
        weights = { a : { o : 1 for o in t.keys() } for a, t in gt.iteritems() }

    correct = {}
    total = {}
    nb_corr = {}
    for k, r_value in results.iteritems() :

        # Check that we are including this key
        if k not in exclude_keys :
            c_methods = {}
            t_methods = {}
            w_norm = weights[k]

            # Calculate sum of correct and total counts
            for method, r_method in r_value.iteritems() :
                r_weighted_correct = [numpy.array(r) * w_norm[o] for o, r in r_method["correct"].iteritems()]
                r_weighted_total = [numpy.array(r) * w_norm[o] for o, r in r_method["total"].iteritems()]
                c_methods[method] = numpy.sum(r_weighted_correct, axis = 0)
                t_methods[method] = numpy.sum(r_weighted_total, axis = 0)

            # Calculate sum of possible correspondences
            nb_corr[k] = int(sum(map(lambda (o,data) : data["nb_correspondences"]*w_norm[o], gt[k].iteritems())))

            # save accumulated counts
            correct[k] = c_methods
            total[k] = t_methods

    return correct, total, nb_corr
