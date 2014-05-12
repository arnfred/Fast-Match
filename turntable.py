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
from itertools import chain
from cache import Metric_Cache
from turntable_ground_truth import get_turntable_path
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

    # Get paths to the three images
    def get_path(i) : return {
            "A" : get_turntable_path(object_type, angles[0] + i*360, "Bottom"),
            "B" : get_turntable_path(object_type, angles[0] + i*360, "Top"),
            "C" : get_turntable_path(object_type, angles[1] + i*360, "Bottom")
        }

    if verbose :
        print("matching\n%s\n%s" %(get_path(0)["A"], get_path(0)["C"]))

    # Get paths
    def get_match_fun(i) :
        # Collect matches
        paths = get_path(i)
        query_path, target_path = paths["A"], paths["C"]
        query_cache, target_img = Metric_Cache(query_path), imaging.open_img(target_path)
        return match_fun(query_cache, target_img, options = options)

    # Get matches
    match_funs = map(get_match_fun, range(3))
    matches_df = pd.DataFrame({ img : { tau : f(tau) for tau in thresholds } for img, f in enumerate(match_funs) })

    # function for counting correct and total
    def get_correct(row) :
        gt = ground_truth["correspondences"][row.name]
        def gt_int(k) : return [map(int,p) for p in gt.get(k, [])]
        def pos_int(v) : return map(int, v["positions"][1])
        def count(matches) : return sum([pos_int(v) in gt_int(k) for k,v in matches.items()])
        return row.map(count)
    correct = matches_df.apply(get_correct).sum(axis=1)

    def get_total(row) :
        gt = ground_truth["correspondences"][row.name]
        def count(matches) : return sum([len(gt.get(k, [])) > 0 for k in matches.keys()])
        return row.map(count)
    total = matches_df.apply(get_total).sum(axis=1)

    # Get accuracy
    precision = correct / total
    recall = correct / ground_truth["nb_correspondences"]


    return { "precision" : precision, "recall" : recall, "correct" : correct, "total" : total }
    #apply_threshold = lambda tau : list(chain(*((i, f(tau)) for i, f in enumerate(match_functions))))
    #matches = [apply_threshold(tau) for tau in thresholds]

    ## Compare with ground truth
    #get_correct = lambda ms, img : sum([1 for p,r,i in ms if is_correct(ground_truth, p, img, i)])
    #get_total = lambda ms, img : sum([1 for p,r,i in ms if is_counted(ground_truth, p, img, i)])

    ## For each set of matches count correct matches
    #correct = numpy.array([get_correct(ms, img) for img, ms in matches])
    #total = numpy.array([get_total(ms, img) for img, ms in matches])
    #accuracy = correct / numpy.array(total, dtype=numpy.float)
    #return { "accuracy" : accuracy, "correct" : correct, "total" : total }


def is_correct(ground_truth, positions, img, index) :
    gt = ground_truth["correspondences"][img].get(index,[])
    gt_int = [map(int,p) for p in gt]
    pos_int = map(int, positions[1])
    print("Is %s in %s?" % (pos_int, gt_int))
    return pos_int in gt_int

def is_counted(ground_truth, positions, img, index) :
    gt = ground_truth["correspondences"][img].get(index,[])
    return len(gt) > 0





def evaluate_objects(match_fun, angles, object_types, thresholds, ground_truth_data = None, options = {}) :
    """ Returns number of correct and total matches of match_fun on objects:
        match_fun : Function (Function that takes a list of paths and returns matches)
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_types : List[String] (the objects pictured on the turntable)
        thresholds : List[Float] (list of ratio thresholds)
        options : Dict (Set of parameters for matching etc)
    """
    # Get correct and total matches for different thresholds at current angle_increment
    def get_results(object_type) :
        gt = None if ground_truth_data == None else ground_truth_data[object_type]
        return evaluate(match_fun, angles, object_type, thresholds, gt, options)


    results = [get_results(o) for o in object_types]
    correct =  { o : r["correct"] for r, o in zip(results, object_types) }
    total =  { o : r["total"] for r, o in zip(results, object_types) }

    return { "correct" : correct, "total" : total }





def get_weights(gt, exclude_keys = []) :
    """ Calculate a set of weights per object and index item (usually angle) """
    nb_corr_accu = { a : sum(map(lambda data : data["nb_correspondences"], gt_item.values())) for a, gt_item in gt.iteritems() if a not in exclude_keys }
    weights = { a : { o : 1.0 / (t["nb_correspondences"] / float(nb_corr_accu[a])) for o, t in gt_item.iteritems() } for a, gt_item in gt.iteritems() if a not in exclude_keys }
    weights_sum = { a : numpy.sum(ow.values()) for a, ow in weights.iteritems() }
    weights_normalize = { a : { o : w / weights_sum[a] for o,w in ow.iteritems() } for a,ow in weights.iteritems() }
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
