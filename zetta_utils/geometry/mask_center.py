from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial import distance
from skimage.morphology import binary_dilation


def centroid(binary_image_array: npt.NDArray) -> npt.NDArray:
    # Find the interior pixels (all pixels with a value of 1),
    # and average to find the centroid
    interior_coords = np.argwhere(binary_image_array == 1)
    result = np.mean(interior_coords, axis=0)
    return result


def _center_pole(binary_image_array: npt.NDArray, interior_coords: npt.NDArray) -> tuple:
    # Find the perimeter pixels
    dilated_image = binary_dilation(binary_image_array)
    perimeter_image = dilated_image ^ binary_image_array  # XOR to find perimeter pixels
    perimeter_coords = np.argwhere(perimeter_image == 1)

    # If no perimeter found (e.g., if shape fills the entire array)
    if len(perimeter_coords) == 0:
        # Create an artificial perimeter around the edge of the array
        h, w = binary_image_array.shape
        artificial_perimeter = []
        for i in range(h):
            artificial_perimeter.append((i, 0))
            artificial_perimeter.append((i, w - 1))
        for j in range(1, w - 1):
            artificial_perimeter.append((0, j))
            artificial_perimeter.append((h - 1, j))
        perimeter_coords_list = artificial_perimeter
    else:
        perimeter_coords_list = [tuple(coord) for coord in perimeter_coords]

    # Convert interior coords to tuples
    interior_coords_list = [tuple(coord) for coord in interior_coords]

    # Calculate the distance between each interior pixel and all perimeter pixels
    distances = distance.cdist(interior_coords_list, perimeter_coords_list, metric="euclidean")

    # Find the minimum distance to a perimeter pixel for each interior pixel
    min_distances = distances.min(axis=1)

    # Identify the interior pixel with the maximum of these minimum distances
    max_min_distance_index = np.argmax(min_distances)
    pole_of_inaccessibility = interior_coords_list[max_min_distance_index]
    return pole_of_inaccessibility


def center_pole(binary_image_array: npt.NDArray) -> tuple:
    interior_coords = np.argwhere(binary_image_array == 1)
    return _center_pole(binary_image_array, interior_coords)


def interior_point(binary_image_array: npt.NDArray) -> Optional[tuple]:
    """
    Return a quick but reliable interior point: use the centroid if
    the given mask actually includes its own centroid, otherwise fall
    back on the (more expensive) center pole.

    If the given array is entirely 0 (false), this returns None.
    """
    interior_coords = np.argwhere(binary_image_array == 1)
    if interior_coords.size == 0:
        return None
    centroid = tuple(  # pylint: disable=redefined-outer-name
        np.round(np.mean(interior_coords, axis=0)).astype(int)
    )
    if binary_image_array[centroid] == 1:
        return centroid
    return _center_pole(binary_image_array, interior_coords)
