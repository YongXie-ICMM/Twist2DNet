# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:59:42 2023

@author: Haitao Yang & Yong Xie
"""
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math


# ------ UTILITY FUNCTIONS ------

# Calculate angle between the line connecting two leftmost points and the x-axis
def get_left_angle(points):
    # Sort points by x-coordinate to get the leftmost points
    sorted_points = sorted(points, key=lambda point: point[0])
    leftmost_point1, leftmost_point2 = sorted_points[:2]
    # Calculate angle in radians
    angle_radians = math.atan2(leftmost_point1[1] - leftmost_point2[1], leftmost_point1[0] - leftmost_point2[0])
    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


# Calculate the vertices of an equilateral triangle given its center and side length
def find_equilateral_triangle_vertex(center, side_length):
    angle = 2 * math.pi / 3
    # Compute the vertices
    vertices = [
        (int(center[0] + side_length * math.cos(i * angle)),
         int(center[1] + side_length * math.sin(i * angle))) for i in range(3)
    ]
    return vertices


# Rotate the triangle's vertices using the given rotation matrix
def get_rotate_triangle(rotation_matrix, points):
    # Rotate each point
    rotated_vertices = [tuple(np.dot(rotation_matrix, np.array([pt[0], pt[1], 1]))[:2]) for pt in points]
    return np.array(rotated_vertices, np.int32)


# Draw an equilateral triangle and also generate a mask for it
def draw_with_mask(draw_image, center, side_length, rotation_angle_degrees, color):
    # Get the rotation matrix for the given angle
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle_degrees, 1)
    # Compute triangle's vertices without rotation
    triangle0 = np.array(find_equilateral_triangle_vertex(center, side_length), np.int32)
    # Apply rotation to get the final vertices
    triangle1 = get_rotate_triangle(rotation_matrix, triangle0)
    # Initialize a mask with zeros having the same dimensions as the image
    mask = np.zeros_like(draw_image)
    # Draw the triangle on the mask
    cv2.fillPoly(mask, [triangle1], (255, 255, 255), cv2.LINE_AA)
    # Draw the triangle on the image
    cv2.fillPoly(draw_image, [triangle1], color, cv2.LINE_AA)
    # Compute the centroid of the triangle
    center1 = np.mean(triangle1, axis=0)
    return draw_image, mask, center1, side_length, triangle1


# Calculate the maximum possible side length for the second triangle given the properties of the first triangle
def get_two_max_length(center_1, side_length_1, rotation_angle_1):
    rotation_angle_2 = abs(rotation_angle_1 - get_left_angle(find_equilateral_triangle_vertex(center_1, side_length_1)))
    two_max_side_length = abs((1 / (math.sqrt(3) * math.sin(math.radians(rotation_angle_2)) + math.cos(
        math.radians(rotation_angle_2)))) * side_length_1)
    two_side_length = np.random.randint(int(two_max_side_length * 1 / 4), int(two_max_side_length))
    return two_side_length


# Get the slope of an equilateral triangle
def get_triangle_slope(points):
    k_a = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])   # (y2-y1)/(x2-x1)
    k_b = (points[2][1] - points[0][1]) / (points[2][0] - points[0][0])
    k_c = (points[2][1] - points[1][1]) / (points[2][0] - points[1][0])
    # Draw line segments and vertices equidistant from the three points based on the slopes
    return k_a, k_b, k_c


def draw_segments_with_equal_slopes(image, points):
    # Get the slopes of the three sides
    k_a, k_b, k_c = get_triangle_slope(points)

    # Calculate the slope of the side opposite to each vertex
    k_opposite_a = -1 / k_a if k_a != 0 else np.inf
    k_opposite_b = -1 / k_b if k_b != 0 else np.inf
    k_opposite_c = -1 / k_c if k_c != 0 else np.inf

    # Draw line segments while maintaining equal slopes opposite to the vertices
    for i in range(len(points)):
        p1 = points[i]

        # Calculate the start and end points of the line segment based on the slope opposite to the vertex
        if i == 0:
            segment_slope = k_opposite_a
        elif i == 1:
            segment_slope = k_opposite_b
        else:
            segment_slope = k_opposite_c

        # Calculate the direction of the line segment
        direction = np.array([1, segment_slope])
        direction /= np.linalg.norm(direction)

        # Calculate the length of the line segment
        distance = np.linalg.norm(np.array(p1) - np.array(p1))

        # Calculate the start and end points of the line segment
        segment_start = p1
        print("segment_start:{}".format(segment_start))
        segment_end = p1 + distance * direction
        print("segment_end:{}".format(segment_end))
        # Draw the line segment on the image
        cv2.line(image, tuple(segment_start), tuple(segment_end), color=(255, 255, 255), thickness=2)

    # Draw the vertices
    for p in points:
        cv2.circle(image, tuple(p), radius=5, color=(255, 255, 255), thickness=-1)


def calculate_points_on_edges(vertex_A, vertex_B, vertex_C, distance):
    side_length = np.sqrt((vertex_B[0] - vertex_A[0]) ** 2 + (vertex_B[1] - vertex_A[1]) ** 2)
    point_offset = int(distance * side_length)

    point_on_AB_1 = ((vertex_A[0] * (side_length - point_offset) + vertex_B[0] * point_offset) // side_length,
                     (vertex_A[1] * (side_length - point_offset) + vertex_B[1] * point_offset) // side_length)

    point_on_AB_2 = ((vertex_B[0] * (side_length - point_offset) + vertex_A[0] * point_offset) // side_length,
                     (vertex_B[1] * (side_length - point_offset) + vertex_A[1] * point_offset) // side_length)

    point_on_BC_1 = ((vertex_B[0] * (side_length - point_offset) + vertex_C[0] * point_offset) // side_length,
                     (vertex_B[1] * (side_length - point_offset) + vertex_C[1] * point_offset) // side_length)

    point_on_BC_2 = ((vertex_C[0] * (side_length - point_offset) + vertex_B[0] * point_offset) // side_length,
                     (vertex_C[1] * (side_length - point_offset) + vertex_B[1] * point_offset) // side_length)

    point_on_CA_1 = ((vertex_C[0] * (side_length - point_offset) + vertex_A[0] * point_offset) // side_length,
                     (vertex_C[1] * (side_length - point_offset) + vertex_A[1] * point_offset) // side_length)

    point_on_CA_2 = ((vertex_A[0] * (side_length - point_offset) + vertex_C[0] * point_offset) // side_length,
                     (vertex_A[1] * (side_length - point_offset) + vertex_C[1] * point_offset) // side_length)

    return [point_on_AB_1, point_on_AB_2, point_on_BC_1, point_on_BC_2, point_on_CA_1, point_on_CA_2]


# ------ MAIN EXECUTION ------

if __name__ == '__main__':
    # Define the path where generated images will be saved
    save_path = "./datasets_path/"
    image_count = 0
    for i in range(100):
        r = np.random.randint(0, 2)
        t = np.random.randint(0, 2)
        # Initialize a blank image
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Generate a random angle for the first triangle
        rotation_angle_1 = np.random.randint(0, 180)

        # Draw the first green triangle and get its properties
        image, mask, center_1, side_length_1, one_points = draw_with_mask(image, (
        np.random.randint(100, 412), np.random.randint(100, 412)), np.random.randint(50, 200), rotation_angle_1,
                                                                           (0, 128, 0))
        image2 = image
        # Calculate a random offset for the second triangle's center
        offset_max = side_length_1 / 8
        center_2 = (
            center_1[0] + np.random.randint(-offset_max, offset_max),
            center_1[1] + np.random.randint(-offset_max, offset_max)
        )

        if r == 1:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            distance = np.random.uniform(7 * (side_length_1 / (3 * side_length_1)) / 8,
                                         side_length_1 / (3 * side_length_1))
            # Apply step size and offset to the samples
            points = calculate_points_on_edges(one_points[0], one_points[1], one_points[2], distance)
            points_array = np.array(points, np.int32)
            points_array = points_array.reshape((-1, 1, 2))
            cv2.fillPoly(image, [points_array], (0, 128, 0), cv2.LINE_AA)

        # Calculate leftmost angle for the first triangle
        one_angle = get_left_angle(one_points)

        # Get maximum side length for the second triangle
        two_side_length = get_two_max_length(center_1, side_length_1, rotation_angle_1)

        # Initialize another blank image for the red triangle
        red_image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Generate a random angle for the second triangle and draw it
        rotation_angle_2 = np.random.randint(0, 180)
        red_image, _, _, _, two_points = draw_with_mask(red_image, center_2, two_side_length, rotation_angle_2,
                                                        (0, 0, 128))
        # Calculate leftmost angle for the second triangle
        two_angle = get_left_angle(two_points)

        if t == 1:
            red_image = np.zeros((512, 512, 3), dtype=np.uint8)
            distance_2 = np.random.uniform(7 * (two_side_length / (3 * two_side_length)) / 8,
                                           two_side_length / (3 * two_side_length))
            # Apply step size and offset to the samples
            points_2 = calculate_points_on_edges(two_points[0], two_points[1], two_points[2], distance_2)
            points_array_2 = np.array(points_2, np.int32)
            points_array_2 = points_array_2.reshape((-1, 1, 2))
            cv2.fillPoly(red_image, [points_array_2], (0, 0, 128), cv2.LINE_AA)

        # Merge the red triangle with the green triangle using the mask of the green triangle
        combined_mask = cv2.bitwise_and(red_image, mask)

        combined_has_color = combined_mask.any(axis=-1)
        image_has_color = image.any(axis=-1)
        update_condition = combined_has_color & image_has_color

        image[update_condition] = combined_mask[update_condition]

        # Determine and adjust the angular difference between the triangles
        if abs(int(two_angle - one_angle)) >= 180:
            last_angle = abs(int(two_angle - one_angle)) - 180
            last_angle = 120 - last_angle if last_angle > 60 else last_angle
        else:
            last_angle = 120 - abs(int(two_angle - one_angle)) if abs(int(two_angle - one_angle)) > 60 else abs(
                int(two_angle - one_angle))
        if last_angle > 30:
            last_angle = 60 - last_angle
        # Save the generated image
        cv2.imwrite(save_path + "image_{}_{}.png".format(last_angle, image_count), image)
        image_count += 1
