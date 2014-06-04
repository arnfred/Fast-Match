"""
Python module for use with openCV to display keypoints
and their links between images

Jonas Toft Arnfred, 2013-04-24
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import pylab
import numpy
import imaging

####################################
#                                  #
#             Images               #
#                                  #
####################################

def append_images(im1, im2, seperator = 0) :
    """ return a new image that appends the two images side-by-side.
    """

    barrier = numpy.ones((im1.shape[0],seperator, 3), dtype=numpy.float)
    im1_float = im1 / 255.0
    im2_float = im2 / 255.0

    if (im1.shape[0] == im2.shape[0]) :
        tmp = im2_float

    elif (im1.shape[0] > im2.shape[0]) :
        tmp = numpy.ones((im1.shape[0], im2.shape[1], 3), dtype=numpy.float)
        tmp[0:im2.shape[0], :, :] = im2_float

    elif (im1.shape[0] < im2.shape[0]) :
        tmp = numpy.ones((im1.shape[0], im2.shape[1], 3), dtype=numpy.float)
        tmp[0:im1.shape[0], :, :] = im2_float[0:im1.shape[0], :, :]

    else :
        print("Detonating thermo-nuclear devices")

    return numpy.concatenate((im1_float, barrier, tmp), axis=1)


def keypoints(im, pos) :
    """ show image with features. input: im (image as array),
        locs (row, col, scale, orientation of each feature)
    """
    # Plot all keypoints
    pylab.imshow(im)
    for i, (x,y) in enumerate(pos) :
        pylab.plot(x, y, marker='.', color = getRedGreen(float(i+1) / len(pos)))
    pylab.axis('off')
    pylab.show()


def compare_keypoints(im1, im2, pos1, pos2, filename = None, separation = 0) :
    """ Show two images next to each other with the keypoints marked
    """

    # Construct unified image
    im3 = append_images(im1,im2, separation)

    # Find the offset and add it
    offset = im1.shape[1]
    pos2_o = [(x+offset + separation,y) for (x,y) in pos2]

    # Create figure
    fig = pylab.figure(frameon=False, figsize=(12.0, 8.0))
    #ax = pylab.Axes(fig, [0., 0., 1., 1.])

    # Show images
    pylab.gray()
    pylab.imshow(im3)
    pylab.plot([x for x,y in pos1], [y for x,y in pos1], marker='o', color = '#00aaff', lw=0)
    pylab.plot([x for x,y in pos2_o], [y for x,y in pos2_o], marker='o', color = '#00aaff', lw=0)
    pylab.axis('off')

    pylab.xlim(0,im3.shape[1])
    pylab.ylim(im3.shape[0],0)

    if filename != None :
        fig.savefig(filename, bbox_inches='tight', dpi=300)


def matches(im1, im2, matches, dist = None, options = {}) :
    """ show a figure with lines joining the accepted matches in im1 and im2
        input: im1,im2 (images as arrays), locs1,locs2 (location of features),
        matchscores (as output from 'match').
    """

    scale    = options.get("scale", 1)
    filename   = options.get("filename", None)
    max_dist   = options.get("max_dist", 100)
    line_width = options.get("line_width", 0.8)
    size       = options.get("size", (12, 8))
    separation = options.get("separation", 20)

    if scale != 1 :
        s = numpy.array([scale, scale])
        im1_size = numpy.array(im1.shape[1::-1] * s, dtype=numpy.uint16)
        im2_size = numpy.array(im2.shape[1::-1] * s, dtype=numpy.uint16)
        if scale < 0.5 :
            im1_scaled = imaging.get_thumbnail(im1, size = im1_size)
            im2_scaled = imaging.get_thumbnail(im2, size = im2_size)
        else :
            im1_scaled = imaging.open_img(im1, size = im1_size)
            im2_scaled = imaging.open_img(im2, size = im2_size)
        im3 = append_images(im1_scaled, im2_scaled, separation)
        matches = [(m[0] * s, m[1] * s) for m in matches]
    else :
        im3 = append_images(im1,im2, separation)
        im1_scaled = im1

    # Create figure
    fig = pylab.figure(frameon=False, figsize=size)
    ax = pylab.Axes(fig, [0., 0., 1., 1.])

    ax.set_axis_off()
    fig.add_axes(ax)

    # Display image
    ax.imshow(im3)

    # Get colors
    if dist != None and len(dist) == len(matches) :
        cs = [getRedGreen(numpy.log(d+1)/numpy.log(max_dist)) for d in dist]
    else :
        cs = ['#00aaff' for m in matches]

    # Plot all lines
    offset_x = im1_scaled.shape[1]
    for i,((x1,y1),(x2,y2)) in enumerate(matches) :
        ax.plot([x1, x2+offset_x + separation], [y1,y2], color=cs[i], lw=line_width)

    pylab.xlim(0,im3.shape[1])
    pylab.ylim(im3.shape[0],0)

    if filename != None :
        fig.savefig(filename, bbox_inches='tight', dpi=72)


def getRedGreen(f) :
    if f > 1 : f = 1
    elif f < 0 : f = 0
    c = '#%02x%02x11' % (int(f*255), int((1-f)*255))
    return c


def visualize_log(log, im1, im2, stop_at = None, scale = None, size = (14, 8), filename = None, alt = False) :
    # Prepare images
    if scale == None :
        scale = min(1.0, 600 / float(im1.shape[1]))
    s = numpy.array([scale, scale])
    im1_size = numpy.array(im1.shape[1::-1] * s, dtype=numpy.uint16)
    im2_size = numpy.array(im2.shape[1::-1] * s, dtype=numpy.uint16)
    im1_scaled = imaging.get_thumbnail(im1, size = im1_size)
    im2_scaled = imaging.get_thumbnail(im2, size = im2_size)
    separation = 20
    offset_x = im1_size[0] + separation
    im3 = append_images(im1_scaled, im2_scaled, separation)

    def translate_point(point, image = "im1") :
        x, y = numpy.array(point) * s
        if image == "im1" :
            return x, y
        else :
            return (x + offset_x, y)

    # Create figure
    fig = pylab.figure(frameon=False, figsize=size)
    ax = pylab.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # Display image
    ax.imshow(im3)

    # Plot log data
    for i, d in enumerate(log) :
        # Enough?
        if stop_at != None and i > stop_at :
            break
        if alt :
            # Plot target grid
            q_cell = d["query_cell"]
            q_block = d["query_block"]
            t_cell = d["target_cell"]
            t_block = d["target_block"]
            #margin = d["margin"]
            # Add squares
            ax.add_patch(pylab.Rectangle( translate_point((q_cell.get_x_min(), q_cell.get_y_min()), "im1"),
                                (q_cell.get_x_max()-q_cell.get_x_min()) * scale, (q_cell.get_y_max()-q_cell.get_y_min()) * scale, color="#0055aa", fill = False))
            ax.add_patch(pylab.Rectangle( translate_point((q_block.get_x_min()+2, q_block.get_y_min()+2), "im1"),
                                (q_block.get_x_max()-q_block.get_x_min()-4) * scale, (q_block.get_y_max()-q_block.get_y_min()-4) * scale, color="#000055", fill = False))
            ax.add_patch(pylab.Rectangle( translate_point((t_cell.get_x_min(), t_cell.get_y_min()), "im2"),
                                (t_cell.get_x_max()-t_cell.get_x_min()) * scale, (t_cell.get_y_max()-t_cell.get_y_min()) * scale, color="#0055aa", fill = False))
            ax.add_patch(pylab.Rectangle( translate_point((t_block.get_x_min()+2, t_block.get_y_min()+2), "im2"),
                                (t_block.get_x_max()-t_block.get_x_min()-4) * scale, (t_block.get_y_max()-t_block.get_y_min()-4) * scale, color="#000055", fill = False))
        else :
            # Plot target grid
            cell = d["target_cell"]
            block = d["target_block"]
            query_pos = d["current_pos"][:2]
            #margin = d["margin"]
            # Add square
            ax.add_patch(pylab.Rectangle( translate_point((cell.get_x_min(), cell.get_y_min()), "im2"),
                                (cell.get_x_max()-cell.get_x_min()) * scale, (cell.get_y_max()-cell.get_y_min()) * scale, color="#0055aa", fill = False))
            ax.add_patch(pylab.Rectangle( translate_point((block.get_x_min()+2, block.get_y_min()+2), "im2"),
                                (block.get_x_max()-block.get_x_min()-4) * scale, (block.get_y_max()-block.get_y_min()-4) * scale, color="#000055", fill = False))
            # Add circle
            ax.add_patch(pylab.Circle( translate_point(query_pos, "im1"), d["radius"] * scale, fill = False, linewidth = 0.5))
            # Add matches
        for match in d["matches"] :
            pos_q = match[:2]
            pos_t = match[2:]
            x1, x2 = numpy.array([pos_q[0], pos_t[0]]) * s
            y1, y2 = numpy.array([pos_q[1], pos_t[1]]) * s
            ax.plot([x1, x2+offset_x], [y1,y2], color="#2595e3", lw=1)

    # Limit figure to area around im3
    pylab.xlim(0,im3.shape[1])
    pylab.ylim(im3.shape[0],0)
    if filename != None :
        fig.savefig("%s_%i.png" % (filename, stop_at), bbox_inches='tight', dpi=72)
