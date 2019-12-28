import numpy as np


def calc_fov(focal_length, sensor_size):
    return 2 * np.arctan(0.5 * sensor_size / focal_length)


def calc_signed_coc(focal_length, focus_distance, f_number, depth):
    return ((focal_length ** 2) * (depth - focus_distance)) /\
           (f_number * depth * (focus_distance - focal_length))


def calc_depth_from_signed_coc(focal_length, focus_distance, f_number, signed_coc):
    return ((focal_length ** 2) * focus_distance) /\
           (focal_length ** 2 - signed_coc * f_number * (focus_distance - focal_length))


def calc_focus_distance(focal_length, f_number, signed_coc, depth):
    return (focal_length * depth * (signed_coc * f_number + focal_length)) / \
           (signed_coc * f_number * depth + focal_length ** 2)


def calc_focus_distance_for_const_range(focal_length, f_number, signed_coc, depth_focus_diff):
    a = 1
    b = depth_focus_diff - focal_length
    c = -depth_focus_diff * focal_length * (1 + (focal_length / (f_number * signed_coc)))

    return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    #return np.roots([a, b, c])


def calc_focus_distance_for_scale_range(focal_length, f_number, signed_coc, focus_multiplier):
    foc_min = focal_length * ((focal_length * (focus_multiplier - 1)) / (focus_multiplier * f_number * signed_coc) + 1)
    foc_max = foc_min * focus_multiplier

    return foc_min, foc_max


