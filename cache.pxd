"""
Header file for cache.pyx

Jonas Toft Arnfred, 2013-05-08
"""
cimport numpy

cdef class Grid_Pos :
    cdef int row, col

cdef class Pos :
    cdef int x, y

cdef class Size :
    cdef int w, h

cdef class Pos_Pair :
    cdef Pos query, target

cdef class Frame :
    cdef int x_min, x_max, y_min, y_max

cdef class Grid_Cache :
    cdef int rows
    cdef int cols
    cdef public int margin
    cdef Size img_size
    cdef int width
    cdef int height
    cdef public Size cell_size
    cdef public Size block_size
    cdef numpy.ndarray data
    cdef public Frame last_cell
    cdef public Frame last_block
    cdef object cells
    cdef object blocks
    # Methods
    cdef Frame frame(self, Grid_Pos pos, Size size)
    cdef object get_cell_data(self, Grid_Pos cell)
    cdef Pos center(self, Grid_Pos cell)
    cdef object cache(self, Grid_Pos cell)
    cdef Grid_Pos block(self, Pos p)
    cdef Grid_Pos cell(self, Pos p)
    cpdef get(self, Pos p)
    cdef object cache_block(self, Grid_Pos cell, Grid_Pos block)
    cpdef object get_neighbor(self, Grid_Pos block, Pos p)

cdef class Metric_Cache :
    cdef public char* path
    cdef public object thumb
    cdef public object original
    # methods
    cdef object load(self, char* dir = ?)
    cdef create_thumbnail(self, char* path, int thumb_x, int thumb_y)
    cdef create_image(self, char* path, int max_size, char* metric)
    cdef object save(self, char* dir = ?)
    cdef object get(self, int x, int y, int radius, object options = ?)


