from copy import deepcopy
import cv2
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance
from shapely.geometry import LineString

EPSILON = 1e-6


def get_closest_point_on_edge(edge_uv_px, point_uv_px):
    """
    Given an edge containing many points and a test point, finds the closest point on the edge
    to that point and the percent along the edge.

    Returns:
        (
            (float, float): the closest point along the edge,
            int: the index of the start point edge segment that is closest,
            float: the percent along the closest edge segment to get to the closest,
            (float): distance from the test point to the closest point
    """
    for i, edge_segment_uv_px in enumerate(zip(edge_uv_px[:-1], edge_uv_px[1:])):
        closest_point_uv_px, percent_distance = get_closest_point_on_edge_segment(edge_segment_uv_px, point_uv_px)
        if percent_distance >= 0 and percent_distance <= 1:
            return (
                closest_point_uv_px,
                i,
                percent_distance,
                math.dist(point_uv_px, closest_point_uv_px),
            )
        elif i == 0 and percent_distance < 0:
            return (
                edge_segment_uv_px[0],
                i,
                0.0,
                math.dist(point_uv_px, edge_segment_uv_px[0]),
            )
        elif i == len(edge_uv_px) - 2 and percent_distance > 1:
            return (
                edge_segment_uv_px[-1],
                i,
                1.0,
                math.dist(point_uv_px, edge_segment_uv_px[-1]),
            )

    distances_px = [math.dist(edge_point_uv_px, point_uv_px) for edge_point_uv_px in edge_uv_px]
    min_index = distances_px.index(min(distances_px))
    return edge_uv_px[min_index], min_index, 0.0, distances_px[min_index]


def get_closest_point_on_edge_segment(edge_segment_uv_px, point_uv_px):
    """
    Given an edge segment consisting of two points, and a test point, returns the closest point
    on the edge segment to the test point.

    Returns:
        (
            (float, float): the closest point to the edge segment,
            float: the percent distance along the edge segment to that point
        )
    """
    [[x1, y1], [x2, y2]], [a, b] = edge_segment_uv_px, point_uv_px
    if x1 == x2:
        return (x1, b)
    if y1 == y2:
        return (a, y1)
    m1 = (y2 - y1) / (x2 - x1)
    m2 = -1 / m1
    x = (m1 * x1 - m2 * a + b - y1) / (m1 - m2)
    y = m2 * (x - a) + b
    return [x, y], get_pct_along_edge_segment(edge_segment_uv_px, [x, y])


def get_pct_along_edge_segment(edge_segment_uv_px, point_uv_px):
    """
    Given a point on a line defined by an edge segment (IMPORTANT!), returns the percent
    along the line segment that the point lies.
    """
    [[x1, y1], [x2, y2]], [x, y] = edge_segment_uv_px, point_uv_px
    edge_length_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist_to_start_px = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    dist_to_end_px = math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    is_neg = dist_to_end_px > dist_to_start_px and dist_to_end_px > edge_length_px
    pct_dist_px = dist_to_start_px / edge_length_px
    return -pct_dist_px if is_neg else pct_dist_px


def get_overlaps(edge1_uv_px, edge2_uv_px):
    overlaps = []
    curr_overlap = None
    for edge2_index, point2_uv_px in enumerate(edge2_uv_px):
        closest_point_uv_px, edge1_index, pct, dist_px = get_closest_point_on_edge(edge1_uv_px, point2_uv_px)
        if dist_px > 0.1:
            if curr_overlap is not None:  # End of overlap
                prev_edge2_segment_uv_px = edge2_uv_px[(edge2_index - 1):(edge2_index + 1)]
                closest_point_uv_px, _, pct, dist_px = get_closest_point_on_edge(
                    prev_edge2_segment_uv_px, edge1_uv_px[edge1_index + 1]
                )
                if (edge1_index + 1) == len(edge1_uv_px) - 1:
                    curr_overlap.append(
                        [
                            closest_point_uv_px,
                            (edge1_index, 1.0),
                            (edge2_index - 1, pct),
                        ]
                    )
                else:
                    curr_overlap.append(
                        [
                            closest_point_uv_px,
                            (edge1_index + 1, 0.0),
                            (edge2_index - 1, pct),
                        ]
                    )
                overlaps.append(curr_overlap)
                curr_overlap = None
            continue
        if curr_overlap is None:  # Beginning of new overlap
            curr_overlap = [closest_point_uv_px, (edge1_index, pct), (edge2_index, 0.0)]
        elif pct != 1.0 or edge2_index == (len(edge2_uv_px) - 1):  # End of overlap
            if edge2_index == (len(edge2_uv_px) - 1):
                curr_overlap.append([closest_point_uv_px, (edge1_index, pct), (edge2_index - 1, 1.0)])
            else:
                curr_overlap.append([closest_point_uv_px, (edge1_index, pct), (edge2_index, 0.0)])
            overlaps.append(curr_overlap)
            curr_overlap = None
    return overlaps


def merge_edges(edges_uv_px):
    width_px = max([point_uv_px[0] for edge_uv_px in edges_uv_px for point_uv_px in edge_uv_px])
    height_px = max([point_uv_px[1] for edge_uv_px in edges_uv_px for point_uv_px in edge_uv_px])
    image = Image.new("L", (math.ceil(width_px * 10.0), math.ceil(height_px * 10.0)))
    image_draw = ImageDraw.Draw(image)
    for edge_uv_px in edges_uv_px:
        image_draw.line([(int(x[0] * 10.0), int(x[1] * 10.0)) for x in edge_uv_px], fill=255)
    image.show()
    contours = []
    for contour in cv2.findContours(np.array(image), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
        contour = [[pt[0] / 10.0, pt[1] / 10.0] for pt in contour.reshape(contour.shape[0], 2).tolist()]
        import pdb

        pdb.set_trace()
        contours.append(np.array(LineString(contour).simplify(1.0).coords.xy).T.tolist()[:-1])

    image = Image.new("L", (math.ceil(width_px), math.ceil(height_px)))
    image_draw = ImageDraw.Draw(image)
    for contour in contours:
        image_draw.line([(int(x[0]), int(x[1])) for x in contour], fill=255, width=1)
    image.show()

    return contours


def min_distance(A, B, E):
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]

    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]

    # vector AP
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # Minimum distance from
    # point E to the line segment
    reqAns = 0

    # Case 1
    if AB_BE > 0:
        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = math.sqrt(x * x + y * y)

    # Case 2
    elif AB_AE < 0:
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = math.sqrt(x * x + y * y)

    # Case 3
    else:
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = math.sqrt(x1 * x1 + y1 * y1)
        reqAns = abs(x1 * y2 - y1 * x2) / mod

    return reqAns


def merge_lines(lines, tolerance=0.125):
    if len(lines) == 0:
        return []
    lines = [line for line in lines if line != []]
    final_line = deepcopy(lines[0])
    for line in lines[1:]:
        for point in line:
            min_dist = np.inf
            min_index = -1
            for i, final_line_seg in enumerate(zip(final_line[:-1], final_line[1:])):
                curr_min_dist = min_distance(final_line_seg[0], final_line_seg[1], point)
                if curr_min_dist < min_dist:
                    min_dist = curr_min_dist
                    min_index = i + 1
            if min_dist < tolerance:
                continue
            dist_to_start = distance.euclidean(final_line[0], point)
            dist_to_end = distance.euclidean(final_line[-1], point)
            if dist_to_start <= min_dist:
                final_line.insert(0, point)
            elif dist_to_end <= min_dist:
                final_line.append(point)
            else:
                final_line.insert(min_index, point)

    if len(final_line) > 2:
        i = 0
        while True:
            if i >= len(final_line) - 2:
                break
            if min_distance(final_line[i], final_line[i + 2], final_line[i + 1]) < tolerance:
                final_line.pop(i + 1)
            else:
                i += 1
    return final_line


def get_image_bounds(image_width_px, image_height_px):
    return [
        [0.0, 0.0],
        [image_width_px - 1.0, 0.0],
        [image_width_px - 1.0, image_height_px - 1.0],
        [0.0, image_height_px - 1.0],
        [0.0, 0.0],
    ]


def is_clockwise(poly):
    """
    Given a list of points returns true if the points are in clockwise order.

    Implementation of the following post:
    https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    Note, in image-frame u cross v points inside, whereas in conventional x cross y points in the opposite direction,
    so unlike that post, when the sum is less than 0 the points are clockwise.

    Args:
        poly (list of list of 2 floats): Representing a 2D polygon.

    Returns:
        (bool): whether the polygon is in clockwise order
    """
    sum = 0
    i = -1  # for one point case
    for i in range(len(poly) - 1):
        sum += (poly[i + 1][0] - poly[i][0]) * (poly[i + 1][1] + poly[i][1])
    sum += (poly[0][0] - poly[i + 1][0]) * (poly[0][1] + poly[i + 1][1])
    return sum <= 0


def compute_intersection_2d(edge1: np.ndarray, edge2: np.ndarray):
    """
    Given two edges, computes the intersection point of the two edges.  Return None if the two edges are parallel
    (i.e., if they don't intersect) or if either edge has no length (i.e., the same point repeated).

    edge1 (iterable of two points with two floats): one edge to compute the intersection of
    edge2 (iterable of two points with two floats): another edge to compute the intersection of
    """
    edge1 = np.array(edge1) if type(edge1) == list else edge1
    edge2 = np.array(edge2) if type(edge2) == list else edge2

    assert edge1.shape[1] == 2 and edge1.shape[0] >= 2, "edge1 must have shape N x 2"
    assert edge2.shape[1] == 2 and edge2.shape[0] >= 2, "edge2 must have shape N x 2"

    edge1_start, edge1_end = edge1[0], edge1[-1]
    edge2_start, edge2_end = edge2[0], edge2[-1]
    if np.all(edge1_start == edge1_end) or np.all(edge2_start == edge2_end):
        return None
    edge1_dims = [edge1_start[0] - edge1_end[0], edge1_start[1] - edge1_end[1]]
    edge2_dims = [edge2_start[0] - edge2_end[0], edge2_start[1] - edge2_end[1]]
    n1 = edge1_start[0] * edge1_end[1] - edge1_start[1] * edge1_end[0]
    n2 = edge2_start[0] * edge2_end[1] - edge2_start[1] * edge2_end[0]
    n3 = edge1_dims[0] * edge2_dims[1] - edge1_dims[1] * edge2_dims[0]
    if math.isclose(n3, 0, abs_tol=EPSILON):
        return None
    return np.array(
        [
            (n1 * edge2_dims[0] - n2 * edge1_dims[0]) * (1.0 / n3),
            (n1 * edge2_dims[1] - n2 * edge1_dims[1]) * (1.0 / n3),
        ]
    )


def remove_adjacent_duplicates(items, check_wraparound=True, epsilon=EPSILON):
    """
    Given a list of ints/floats, removes adjacent duplicates
    (e.g., [0, 1, 2, 2, 3, 3, 5, 4, 3, 3, 1, 1] -> [0, 1, 2, 3, 5, 4, 3, 1]
           [[1,2],[1,2],[3,3],[3,4]] -> [[1, 2], [3, 3], [3, 4]]
           [[1, 2, 2.000000001], [1, 2, 2], [3, 3, 3], [3, 4, 5]] -> [[1, 2, 2], [3, 3, 3], [3, 4, 5]]
    )

    Args:
        items (list): a list of ints/floats or a list of list of ints/floats
        check_wraparound (bool): whether to remove last point if it's equal to the first point
        epsilon (float): tolerance for considering an element equal to its neighbor

    Returns:
        (list): a list with all adjacent duplicates removed
    """

    def get_dist_sq(from_index, to_index):
        dist_sq = 0
        if isinstance(items[from_index], list):
            for j in range(len(items[from_index])):
                dist_sq += (items[to_index][j] - items[from_index][j]) ** 2
        else:
            dist_sq += (items[to_index] - items[from_index]) ** 2
        return dist_sq

    if isinstance(items, np.ndarray):
        items = items.tolist()
    for i in range(len(items)):
        if isinstance(items[i], np.ndarray):
            items[i] = items[i].tolist()

    epsilon_sq = epsilon**2
    if check_wraparound and len(items) > 0:
        if get_dist_sq(-1, 0) < epsilon_sq:
            items.pop(-1)

    i = 0
    while i < (len(items) - 1):
        if get_dist_sq(i, i + 1) < epsilon_sq:
            items.pop(i)
        else:
            i += 1
    return items


def in_edge_divider(clip_edge: np.ndarray, subject_point: np.ndarray):
    """
    Given a clip edge from a polygon in which the edge is in clockwise order.

    If the subject_point is to the right of the edge, it's outside the edge.
    If it's to the left of the edge, it's inside the edge.

    Args:
        clip_edge (iterable with two points each with two floats): start point and end point of a clip edge
                                                                   in clockwise order
        subject_point (iterable with two floats): a point to check if it's inside the edge divider (to the left),
                                                  or if it's outside the edge divider (to the right)
    """
    clip_edge = np.array(clip_edge) if type(clip_edge) == list else clip_edge
    subject_point = np.array(subject_point) if type(subject_point) == list else subject_point

    assert clip_edge.shape == (2, 2), "clip_edge must have shape 2 x 2"
    assert subject_point.shape[0] == 2, "subject_point must have shape 2 x 1"

    clip_point_start, clip_point_end = clip_edge[0], clip_edge[1]
    return (clip_point_end[0] - clip_point_start[0]) * (subject_point[1] - clip_point_start[1]) >= (
        clip_point_end[1] - clip_point_start[1]
    ) * (subject_point[0] - clip_point_start[0])


def clip_line_2d(clip_poly: np.ndarray, subject_line: np.ndarray):
    """
    Implemenation of Sutherland-Hodgman in 2D. Polygon should be open.
    https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm

    Args:
        clip_poly (np.ndarray): N x 2 list of (u, v) pixel coordinates of the polygon to clip to;
                                must be clockwise!
        subject_line (np.ndarray): N x 2 list of (u, v) pixel coordinates of the line to clip.

    Returns:
        (np.ndarray): N x 2 list of points of the subject_line clipped to the clip_poly
        (bool): whether the clip_poly clipped any portion of the subject_line
    """
    clip_poly = np.array(clip_poly) if type(clip_poly) == list else clip_poly
    subject_line = np.array(subject_line) if type(subject_line) == list else subject_line

    assert len(clip_poly.shape) == 2 and clip_poly.shape[1] == 2, "Clip poly must have shape N x 2"
    assert len(subject_line.shape) == 2 and subject_line.shape[1] == 2, "Subject line must have shape N x 2"
    assert is_clockwise(clip_poly), "Clip poly must be provided in clockwise order"

    is_clipped = False
    output_list = subject_line

    for clip_point_index in range(len(clip_poly)):
        clip_point_start = clip_poly[(clip_point_index) % len(clip_poly)]
        clip_point_end = clip_poly[(clip_point_index + 1) % len(clip_poly)]
        clip_edge = [clip_point_start, clip_point_end]
        input_list = remove_adjacent_duplicates(output_list)
        input_list = output_list
        output_list = []

        for subject_point_index in range(len(input_list) - 1):
            next_point = input_list[(subject_point_index + 1) % len(input_list)]
            cur_point = input_list[subject_point_index % len(input_list)]
            subject_edge = [cur_point, next_point]
            intersection_px = compute_intersection_2d(clip_edge, subject_edge)

            if in_edge_divider(clip_edge, cur_point):
                output_list.append(cur_point)
                if intersection_px is not None and not in_edge_divider(clip_edge, next_point):
                    output_list.append(intersection_px)
                    is_clipped = True
            elif intersection_px is not None and in_edge_divider(clip_edge, next_point):
                output_list.append(intersection_px)
        if in_edge_divider(clip_edge, next_point):
            output_list.append(next_point)
        if len(output_list) == 0:
            break

    is_clipped = True if len(output_list) == 0 else is_clipped
    output_list = np.array(remove_adjacent_duplicates(output_list))

    return (output_list, is_clipped)


def bound_line(line, image_width_px, image_height_px):
    if len(line) == 0:
        return line
    image_bounds = get_image_bounds(image_width_px, image_height_px)
    return clip_line_2d(image_bounds, line)[0].tolist()


def snap_to_image_boundary(uv_px, image_width_px, image_height_px, tolerance=1e-6, inside_and_out=False):
    """
    Given a pixel that's within tolerance of the image border, set it to the image border.

    Parameters:
        uv_px (list of px points or a single px point): the pixels to bound to the image,
        image_width_px (float or int): the width of the image in pixels,
        image_height_px (float or int): the height of the image in pixels,
        tolerance (float): the tolerance to put bounds around the image border,
        inside_and_out (float): when True, if within tolerance of border whether inside or out, bounds
                                pixel to the image border.
                                when False, only bounds to image border if within tolerance of border
                                outside of the border.

    Returns:
        (list of px points or a single px point): the pixel(s) bounded to the image border
    """
    if len(uv_px) == 0:
        return []
    if type(uv_px[0]) == list:
        return [
            snap_to_image_boundary(
                px,
                image_width_px,
                image_height_px,
                tolerance=tolerance,
                inside_and_out=inside_and_out,
            )
            for px in uv_px
        ]
    uv_px[0] = 0.0 if uv_px[0] < 0 and uv_px[0] >= -tolerance else uv_px[0]
    if inside_and_out:
        in_width_border = uv_px[0] > (image_width_px - 1.0 - tolerance) and uv_px[0] <= (
            image_width_px - 1.0 + tolerance
        )
    else:
        in_width_border = uv_px[0] > (image_width_px - 1.0) and uv_px[0] <= (image_width_px - 1.0 + tolerance)
    uv_px[0] = image_width_px - 1.0 if in_width_border else uv_px[0]
    uv_px[1] = 0.0 if uv_px[1] < 0 and uv_px[1] >= -tolerance else uv_px[1]
    if inside_and_out:
        in_height_border = uv_px[1] > (image_height_px - 1.0 - tolerance) and uv_px[1] <= (
            image_height_px - 1.0 + tolerance
        )
    else:
        in_height_border = uv_px[1] > (image_height_px - 1.0) and uv_px[1] <= (image_height_px - 1.0 + tolerance)
    uv_px[1] = image_height_px - 1.0 if in_height_border else uv_px[1]
    return uv_px
