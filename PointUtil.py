from math import sqrt, acos, degrees
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

mp_hands = mp.solutions.hands


def convert_to_pixels(point, frame):
    h, w, d = frame.shape

    # , int(point.z * d)
    return (int(point.x * w), int(point.y * h))


# def get_distance(l0, l1) -> float:
#     return sqrt (
#         (l0.x - l1.x) ** 2 +
#         (l0.y - l1.y) ** 2 +
#         (l0.z - l1.z) ** 2
#     )

def get_distance(l0, l1) -> float:
    if len(l0) == 2:
        return sqrt(
            (l0[0] - l1[0]) ** 2 +
            (l0[1] - l1[1]) ** 2
        )
    else:
        return sqrt(
            (l0[0] - l1[0]) ** 2 +
            (l0[1] - l1[1]) ** 2 +
            (l0[2] - l1[2]) ** 2
        )


def get_mid_point(a, b):
    result = [0, 0]
    # for i in range(2):
    result[0] = int(a[0] + (b[0] - a[0]) / 2)
    result[1] = int(a[1] + (b[1] - a[1]) / 2)

    return tuple(result)


def lerp(a, b, t):
    return (1 - t) * a + t * b


def value_to_color(value, cmap_name='plasma '):
    """
    Map a float to a color on a gradient.

    Args:
        value (float): The input value to map.
        vmin (float): Minimum value of the range.
        vmax (float): Maximum value of the range.
        cmap_name (str): Colormap name (e.g., 'viridis', 'plasma', etc.).

    Returns:
        str: A color in HEX format.
    """

    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Normalize the value to the range [0, 1]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    normalized_value = norm(value)

    # Map the normalized value to a color
    rgba = cmap(normalized_value)

    # print(rgba)
    return (rgba[0] * 255, rgba[1] * 255, rgba[2] * 255)

    # # Convert RGBA to HEX
    # hex_color = mcolors.to_hex(rgba)
    # return hex_color

def get_angle_between_points(end1: NormalizedLandmark, end2: NormalizedLandmark, center: NormalizedLandmark):
    v1 = (end1.x - center.x, end1.y - center.y, end1.z - center.z)
    v2 = (end2.x - center.x, end2.y - center.y, end2.z - center.z)

    #normalize the vectors
    v1_magnitude = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    v2_magnitude = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])

    v1_normalized = (v1[0]/v1_magnitude, v1[1]/v1_magnitude, v1[2]/v1_magnitude)
    v2_normalized = (v2[0]/v2_magnitude, v2[1]/v2_magnitude, v2[2]/v2_magnitude)

    # Compute the dot product
    dot_product = (v1_normalized[0] * v2_normalized[0] +
                   v1_normalized[1] * v2_normalized[1] +
                   v1_normalized[2] * v2_normalized[2])

    # Ensure the dot product is within the range [-1, 1] to handle floating-point precision errors
    dot_product = max(min(dot_product, 1.0), -1.0)

    # Calculate the angle in radians and convert to degrees
    angle = degrees(acos(dot_product))

    return angle