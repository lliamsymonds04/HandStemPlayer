from math import sqrt
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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