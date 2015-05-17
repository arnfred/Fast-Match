# Fast matching algorithm for sift feature points

## Abstract

Today’s cameras produce images that often exceed 10 megapix
els. Yet computing and matching local features for images
of this size can easily take 20 seconds or more using optimized matching algorithms. This is much too slow for
interactive applications and much too expensive for large
scale image operations. We introduce Fast-Match, an algorithm designed to match large images efficiently without
compromising matching accuracy. It derives its speed from
only computing features in those parts of the image that can
be confidently matched. Fast-Match is an order of magnitude faster than the popular Ratio-Match, yet often doubles
matching precision for difficult image pairs.

## How to install

Just git clone it

## Code example

Run the following code in the main directory

```python
import cv2
from cache import Metric_Cache, Grid_Cache
import fastmatch
import figures

# Open images and create cache
target_path = "images/graf/img1.ppm"
query_path = "images/graf/img4.ppm"
query_cache = Metric_Cache(query_path)
target_img = cv2.imread(target_path)
query_img = cv2.imread(query_path)

# Match using Fast-Match with a ratio threshold of 0.7
log = []
match_fun = fastmatch.match(query_cache, target_img, options = {'log' : log})
matches = list(match_fun(0.7))

# Visualize results
# The [:,:,::-1] inverses the color channels so the images are printed correctly
figures.visualize_log(log, query_img[:,:,::-1], target_img[:,:,::-1], stop_at = 100)
```

## License

The code is released under the [MIT license](http://opensource.org/licenses/MIT).

If you make use of this code, please cite the Fast-Match paper at ICIP 2015.
