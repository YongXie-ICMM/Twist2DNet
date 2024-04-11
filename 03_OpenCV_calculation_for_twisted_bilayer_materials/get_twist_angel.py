import time
import math
import cv2
import torch
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Predict MoS2 twist angle using OpenCV")
    # Exclude background
    parser.add_argument("--image-path", default="./example.png", help="Path to the image for prediction")
    # List of class names
    thickness_dict = ['1L', '2L', 'TL']
    parser.add_argument("--thickness-dict", default=thickness_dict,
                        help="List of class names for plotting the predicted image RGB values")
    parser.add_argument("--result-image", default=False, type=bool,
                        help="Specify whether to draw labels on the original image or segmentation image")
    parser.add_argument("--result-type", default=True, type=bool,
                        help="Specify whether to sketch the angle category or illustrate the label class type")
    parser.add_argument("--get-layers", default=False, type=bool,
                        help="Specify whether to retrieve every layer of the predicted result image")
    parser.add_argument("--draw-interior", default=True, type=bool,
                        help="Specify whether to draw the internal angle of the triangle on the result diagram")
    parser.add_argument("--predict-all-triangle", default=False, type=bool,
                        help="Specify whether to detect all triangles or torsional angle information")
    args = parser.parse_args()
    return args


# get the leftmost two points, the input is the point set of the triangle
def get_left_point(points):
    if points[1][0] < points[0][0]:
        if points[1][0] < points[2][0]:
            point_left1 = points[1]
            point_left2 = points[2] if (points[2][0] < points[0][0]) else points[0]
        else:
            point_left1 = points[2]
            point_left2 = points[1]
    else:
        if points[0][0] < points[2][0]:
            point_left1 = points[0]
            point_left2 = points[2] if (points[2][0] < points[1][0]) else points[1]
        else:
            point_left1 = points[2]
            point_left2 = points[0]
    return point_left1, point_left2


# draw the interior angles of each fitting triangle and display them on the result image
def draw_interior_angle(draw_img, points):
    angle_a1 = math.atan2(-(points[2][1] - points[0][1]),
                          (points[2][0] - points[0][0])) * 180.0 / np.pi
    angle_b1 = math.atan2(-(points[1][1] - points[0][1]),
                          (points[1][0] - points[0][0])) * 180.0 / np.pi

    angle_a2 = math.atan2(-(points[0][1] - points[1][1]),
                          (points[0][0] - points[1][0])) * 180.0 / np.pi
    angle_b2 = math.atan2(-(points[2][1] - points[1][1]),
                          (points[2][0] - points[1][0])) * 180.0 / np.pi

    angle_a3 = math.atan2(-(points[1][1] - points[2][1]),
                          (points[1][0] - points[2][0])) * 180.0 / np.pi
    angle_b3 = math.atan2(-(points[0][1] - points[2][1]),
                          (points[0][0] - points[2][0])) * 180.0 / np.pi
    # calculate the angle of the major axis
    angle1 = (-angle_a1) if (angle_a1 <= 0) else (360 - angle_a1)
    angle2 = (-angle_a2) if (angle_a2 <= 0) else (360 - angle_a2)
    angle3 = (-angle_a3) if (angle_a3 <= 0) else (360 - angle_a3)
    # calculate the end angle of the arc
    end_angle1 = (angle_a1 - angle_b1) if (angle_b1 < angle_a1) else (
            360 + (angle_a1 - angle_b1))
    end_angle2 = (angle_a2 - angle_b2) if (angle_b2 < angle_a2) else (
            360 + (angle_a2 - angle_b2))
    end_angle3 = (angle_a3 - angle_b3) if (angle_b3 < angle_a3) else (
            360 + (angle_a3 - angle_b3))
    cv2.ellipse(draw_img, (int(points[0][0]), int(points[0][1])), (9, 9), angle1,
                0, end_angle1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(draw_img, (int(points[1][0]), int(points[1][1])), (9, 9), angle2,
                0, end_angle2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(draw_img, (int(points[2][0]), int(points[2][1])), (9, 9), angle3,
                0, end_angle3, (255, 0, 0), 2, cv2.LINE_AA)
    # draw the size of each angle on the image
    a1 = round(end_angle1, 2)
    a2 = round(end_angle2, 2)
    a3 = round(end_angle3, 2)
    cv2.putText(draw_img, str(a1),
                (int(points[0][0] - 5), int(points[0][1] - 5)),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(draw_img, str(a2),
                (int(points[1][0]) - 5, int(points[1][1]) - 5),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(draw_img, str(a3),
                (int(points[2][0]) - 5, int(points[2][1]) - 5),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return draw_img


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# Compute the length of a line segment between two points
def get_length(point_x, point_y):
    value = np.sqrt((point_x[0]-point_y[0])**2 + (point_x[1]-point_y[1])**2)
    return value


# obtain angle information for fitting a triangle
def get_triangle_info(points):
    # Obtain the lengths of each side of the triangle
    a = get_length(points[0], points[1])
    b = get_length(points[1], points[2])
    d = get_length(points[2], points[0])
    # Compute the angles using the Law of Cosines: cosA=(b^2+c^2-a^2)/2bc;cosB=(a^2+c^2-b^2)/2ac;cosC=(b^2+a^2-c^2)/2ab
    interior_a = math.acos((b * b + d * d - a * a) / (2 * d * b)) * 180.0 / np.pi
    interior_b = math.acos((a * a + d * d - b * b) / (2 * a * d)) * 180.0 / np.pi
    interior_c = math.acos((a * a - d * d + b * b) / (2 * a * b)) * 180.0 / np.pi
    return interior_a, interior_b, interior_c


# Obtain the twist angle value
def get_twist_angle(args):

    # Detect the twist angle based on the predicted semantic segmentation result image "test_result.png"
    mask_path = args.image_path
    origin_path = './example_original_image.png'
    img = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(mask_path)
    img = cv2.resize(img, (512, 512))
    img2 = cv2.resize(img2, (512, 512))

    # Draw on the original image to obtain the result
    if args.result_image:
        result_img = img
    else:
        result_img = img2
    # Set the color extraction range

    # Background
    lower_index5 = np.array([0, 0, 0])
    upper_index5 = np.array([0, 0, 0])
    color_0 = [lower_index5, upper_index5]

    # The color ranges for various categories are referenced from the 'palette.json' file in the semantic
    # segmentation model.

    # First class color
    lower_index2 = np.array([0, 120, 0])
    upper_index2 = np.array([0, 130, 0])
    color_1 = [lower_index2, upper_index2]
    # Second class color
    lower_index = np.array([0, 0, 120])
    upper_index = np.array([0, 0, 130])
    color_2 = [lower_index, upper_index]
    # Third class color
    lower_index3 = np.array([0, 120, 120])
    upper_index3 = np.array([0, 130, 130])
    color_3 = [lower_index3, upper_index3]

    # Extract spatial distributions of different color ranges
    range_color = [color_1, color_2, color_3]
    test_img = np.zeros((512, 512, 3), np.uint8)
    i = 0
    box_color = [(0, 150, 255), (0, 220, 200), (255, 150, 155)]
    color_dict = [(0, 150, 255), (0, 220, 200), (255, 150, 155)]
    contour_image = np.zeros((512, 512, 3), np.uint8)
    for co in range_color:

        # Perform morphological operations
        mask = cv2.inRange(img2, co[0], co[1])
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_layer_3 = cv2.merge([binary, binary, binary])
        roi_mask = cv2.bitwise_and(img2, mask_layer_3)
        roi_mask_array = np.array(roi_mask)

        # Change the black background to white when saving the images for each layer
        if args.get_layers:
            for h_i in range(img.shape[1] - 1):
                for j in range(img.shape[1] - 1):
                    if ((np.array(roi_mask_array[h_i, j])[0].all() == np.array([0, 0, 0])[0].all()) and (
                            np.array(roi_mask_array[h_i, j])[1].all() == np.array([0, 0, 0])[1].all()) and (
                            np.array(roi_mask_array[h_i, j])[2].all() == np.array([0, 0, 0])[2].all())):
                        roi_mask_array[h_i, j] = [255, 255, 255]
            cv2.imwrite('layers_{}.png'.format(i + 1), roi_mask_array)
        bbox_list = []
        # Use contour detection to get the outer contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t = 0
        for c in contours:
            test_img2 = np.zeros((512, 512, 3), np.uint8)
            # Calculate the centroid of the contour
            m = cv2.moments(c)
            # Check the area of the contour. If it is smaller than a specified threshold, do not process it
            min_area = 15
            if min_area < cv2.contourArea(c):
                # The distance parameter for polygon fitting, which will be used in the next function. For details,
                # please refer to the link in the code
                epsilon = 0.04 * cv2.arcLength(c, True)
                # Approximate the contour by fitting it to a polygon
                approx = cv2.approxPolyDP(c, epsilon, True)
                # Get the number of corners of the polygon
                corners = len(approx)

                # If the `result_type` flag is set to True:
                if args.result_type:
                    # Check the number of corners
                    if corners == 3 or corners == 4:
                        # Use the `minEnclosingTriangle()` function to get the area and the coordinates of the
                        # three vertices of the triangle

                        area, point_triangle1 = cv2.minEnclosingTriangle(c)
                        point_triangle1 = np.squeeze(point_triangle1)
                        # Add a condition to ensure that the internal angle of the triangle is within a
                        # certain range before proceeding
                        min_angle = 55.0
                        max_angle = 65.0
                        test_list = [co[0], co[1]]
                        # Get the length of the sides and the angles of the triangle
                        interior_a, interior_b, interior_c = get_triangle_info(point_triangle1)

                        # If you want to output all fitted triangles and draw the rotation angles on the result image,
                        # choose True. Otherwise, just detect the twist information.

                        # If the `predict_all_triangle` flag is set to True and the triangle satisfies the
                        # angle condition:
                        if args.predict_all_triangle:
                            if min_angle < interior_a and min_angle < interior_b \
                                    and min_angle < interior_c and interior_a < max_angle and interior_b < max_angle \
                                    and interior_c < max_angle:
                                # Draw lines between the three vertices of the fitted triangle
                                cv2.line(result_img, (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                         (int(point_triangle1[1][0]), int(point_triangle1[1][1])), (165, 215, 235), 2,
                                         cv2.LINE_8)
                                cv2.line(result_img, (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1])), (165, 215, 235), 2,
                                         cv2.LINE_8)
                                cv2.line(result_img, (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1])), (165, 215, 235), 2,
                                         cv2.LINE_8)

                                # If the `draw_interior` flag is set to True, draw the interior angles of the fitted
                                # triangle on the result image
                                if args.draw_interior:
                                    result_img = draw_interior_angle(result_img, point_triangle1)

                                # Get the two leftmost points of the fitted triangle
                                point_left1, point_left2 = get_left_point(point_triangle1)

                                # Calculate the rotation angle of the fitted MoS2 triangle relative to the
                                # center of the image
                                rotate_angle_all = math.atan2(-(point_left1[1] - point_left2[1]),
                                                              (point_left1[0] - point_left2[0])) * 180.0 / np.pi
                                rotate_angle_all = rotate_angle_all + 180
                                if rotate_angle_all > 180:
                                    rotate_angle_all = rotate_angle_all - 180

                                # Round the rotation angle to two decimal places and draw it on the image
                                rotate_angle_all = round(rotate_angle_all, 2)
                                cv2.putText(result_img, str(rotate_angle_all),
                                            (int(point_left1[0] + 20), int(point_left1[1] - 10)),
                                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (192, 211, 145), 1, cv2.LINE_AA)

                                # Draw the bounding box of the fitted triangle
                                x, y, w, h = cv2.boundingRect(c)
                                if w > 1 and w > 1:  # suppress small bounding boxes (optional)
                                    bbox_list.extend([x + 1, y + 1, w, h])
                                    cv2.rectangle(result_img, (x, y), (x + w - 1, y + h - 1), box_color[i], 2,
                                                  cv2.LINE_AA)
                        else:
                            if (((np.array(test_list))[0][0].all() == (np.array(color_1))[0][0].all()) and (
                                    (np.array(test_list))[0][1].all() == (np.array(color_1))[0][1].all())
                                    and ((np.array(test_list))[0][2].all() == (np.array(color_1))[0][2].all())):
                                if min_angle < interior_a and min_angle < interior_b \
                                        and min_angle < interior_c and interior_a < max_angle and \
                                        interior_b < max_angle and interior_c < max_angle:
                                    cv2.drawContours(test_img2, [np.array(
                                        [(int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                         (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1]))])], 0,
                                                     (255, 255, 255), -1)

                                    # Get the two leftmost points of the MoS2 fitting triangle and get the rotation
                                    # angle of 1L relative to the center of the image
                                    point_left1_1, point_left2_1 = get_left_point(point_triangle1)
                                    rotate_angle_1 = math.atan2(-(point_left1_1[1] - point_left2_1[1]),
                                                                (point_left1_1[0] - point_left2_1[0])) * 180.0 / np.pi
                                    rotate_angle_1 = rotate_angle_1 if(rotate_angle_1 > 0) else (rotate_angle_1+180)
                                    rotate_angle_1 = round(rotate_angle_1, 2)
                                    mask_img = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)
                                    mask_layer_3 = cv2.merge([mask_img, mask_img, mask_img])
                                    temp_img = cv2.bitwise_and(img2, mask_layer_3)
                                    t = t + 1

                                    # Use inRange function again to extract spatial distribution map of second layer
                                    test1 = cv2.inRange(temp_img, color_2[0], color_2[1])
                                    kernel_test = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                    binary = cv2.morphologyEx(test1, cv2.MORPH_DILATE, kernel_test, iterations=1)
                                    contours1, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                                            cv2.CHAIN_APPROX_SIMPLE)
                                    for ci in contours1:
                                        # Find centroid
                                        m = cv2.moments(ci)
                                        # Check contour area
                                        if min_area < cv2.contourArea(ci):
                                            # Get the perimeter of the contour through the arcLength function and
                                            # pass it to the polygon fitting function to get the
                                            # contour vertex information
                                            epsilon = 0.04 * cv2.arcLength(ci, True)
                                            approx = cv2.approxPolyDP(ci, epsilon, True)

                                            # Get the number of vertices and judge whether it is a triangle
                                            # with three vertices
                                            corners = len(approx)
                                            if corners == 3 or corners == 4:
                                                area, point_triangle = cv2.minEnclosingTriangle(ci)
                                                point_triangle = np.squeeze(point_triangle)

                                                # Get the internal angle information of the fitted triangle
                                                interior_a, interior_b, interior_c = get_triangle_info(point_triangle)
                                                if min_angle < interior_a and \
                                                        min_angle < interior_b and min_angle < interior_c and \
                                                        interior_a < max_angle and interior_b < max_angle and \
                                                        interior_c < max_angle:
                                                    cv2.drawContours(test_img, [np.array(
                                                        [(int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                                         (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1]))])], 0,
                                                                     (0, 0, 255), -1)
                                                    cv2.drawContours(test_img, [np.array(
                                                        [(int(point_triangle[0][0]), int(point_triangle[0][1])),
                                                         (int(point_triangle[1][0]), int(point_triangle[1][1])),
                                                         (int(point_triangle[2][0]), int(point_triangle[2][1]))])], 0,
                                                                     (255, 0, 255), -1)

                                                    for h_i in range(img.shape[1] - 1):
                                                        for j in range(img.shape[1] - 1):
                                                            if ((np.array(test_img[h_i, j])[0].all() ==
                                                                 np.array([0, 0, 0])[
                                                                     0].all()) and (
                                                                    np.array(test_img[h_i, j])[1].all() ==
                                                                    np.array([0, 0, 0])[
                                                                        1].all()) and (
                                                                    np.array(test_img[h_i, j])[2].all() ==
                                                                    np.array([0, 0, 0])[
                                                                        2].all())):
                                                                test_img[h_i, j] = [255, 255, 255]
                                                    # Draw the boundary lines of the fitted triangles of the two layers
                                                    cv2.line(result_img,
                                                             (int(point_triangle[0][0]), int(point_triangle[0][1])),
                                                             (int(point_triangle[1][0]), int(point_triangle[1][1])),
                                                             (192, 211, 145), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                             (int(point_triangle[0][0]), int(point_triangle[0][1])),
                                                             (int(point_triangle[2][0]), int(point_triangle[2][1])),
                                                             (192, 211, 145), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                             (int(point_triangle[1][0]), int(point_triangle[1][1])),
                                                             (int(point_triangle[2][0]), int(point_triangle[2][1])),
                                                             (192, 211, 145), 2, cv2.LINE_8)

                                                    # Draw the boundary lines of the fitted triangle of the single layer
                                                    cv2.line(result_img,
                                                             (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                                             (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                                             (165, 215, 235), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                             (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                                             (int(point_triangle1[2][0]), int(point_triangle1[2][1])),
                                                             (165, 215, 235), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                             (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                                             (int(point_triangle1[2][0]), int(point_triangle1[2][1])),
                                                             (165, 215, 235), 2, cv2.LINE_8)

                                                    # Draw the interior angle of the fitted triangle on the result image
                                                    if args.draw_interior:
                                                        result_img = draw_interior_angle(result_img, point_triangle)
                                                        result_img = draw_interior_angle(result_img, point_triangle1)
                                                    # Get the left two points of the triangle
                                                    point_left1_2, point_left2_2 = get_left_point(point_triangle)
                                                    print("point1:{}  point2:{}".format(point_left1_2, point_left2_2))
                                                    rotate_angle_2 = math.atan2(-(point_left1_2[1] - point_left2_2[1]),
                                                                                (point_left1_2[0] - point_left2_2[
                                                                                    0])) * 180.0 / np.pi
                                                    rotate_angle_2 = rotate_angle_2 if (rotate_angle_2 > 0) else (
                                                                rotate_angle_2 + 180)
                                                    print("rotate1:{}".format(rotate_angle_1))
                                                    print("rotate2:{}".format(rotate_angle_2))
                                                    # Draw the twisting angle
                                                    rotate_angle_2 = np.abs(rotate_angle_1 - rotate_angle_2)
                                                    if rotate_angle_2 > 60:
                                                        rotate_angle_2 = 120 - rotate_angle_2
                                                    rotate_angle_2 = round(rotate_angle_2, 2)
                                                    cv2.putText(result_img, str(rotate_angle_2),
                                                                (int(point_left1_2[0] + 20), int(point_left1_2[1] - 10)),
                                                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (100, 255, 255), 1,
                                                                cv2.LINE_AA)
                else:
                    x, y, w, h = cv2.boundingRect(c)
                    if w > 1 and w > 1:  # suppress small bounding boxes (optional)
                        bbox_list.extend([x + 1, y + 1, w, h])
                        cv2.rectangle(result_img, (x, y), (x + w - 1, y + h - 1), box_color[i], 2, cv2.LINE_8)
                        cv2.putText(result_img, args.thickness_dict[i], ((x + 15), y + 20), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.8,
                                    color_dict[i], 1, cv2.LINE_8)
        i += 1
    cv2.imwrite("example_result.bmp", result_img)


def main():
    args = parse_args()
    t_start = time_synchronized()
    get_twist_angle(args)
    t_end = time_synchronized()
    print("inference+NMS time: {}".format(t_end - t_start))


if __name__ == '__main__':
    main()