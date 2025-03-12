import numpy as np
from scipy.spatial import distance
from skimage.morphology import binary_dilation

def centroid(binary_image_array):
    # Find the interior pixels (all pixels with a value of 1),
    # and average to find the centroid
    interior_coords = np.argwhere(binary_image_array == 1)
    centroid = np.mean(interior_coords, axis=0)
    return centroid

def _center_pole(binary_image_array, interior_coords):
    # Find the perimeter pixels
    dilated_image = binary_dilation(binary_image_array)
    perimeter_image = dilated_image ^ binary_image_array  # XOR to find perimeter pixels
    perimeter_coords = np.argwhere(perimeter_image == 1)

    # Convert the coordinates to lists of tuples (needed by cdist)
    interior_coords = [tuple(coord) for coord in interior_coords]
    perimeter_coords = [tuple(coord) for coord in perimeter_coords]

    # Calculate the distance between each interior pixel and all perimeter pixels
    distances = distance.cdist(interior_coords, perimeter_coords, metric='euclidean')
    
    # Find the minimum distance to a perimeter pixel for each interior pixel
    min_distances = distances.min(axis=1)
    
    # Identify the interior pixel with the maximum of these minimum distances
    max_min_distance_index = np.argmax(min_distances)
    pole_of_inaccessibility = interior_coords[max_min_distance_index]
    return pole_of_inaccessibility

def center_pole(binary_image_array):
    interior_coords = np.argwhere(binary_image_array == 1)
    return _center_pole(binary_image_array, interior_coords)


def interior_point(binary_image_array):
    """
    Return a quick but reliable interior point: use the centroid if
    the given mask actually includes its own centroid, otherwise fall
    back on the (more expensive) center pole.

    If the given array is entirely 0 (false), this returns None.
    """
    interior_coords = np.argwhere(binary_image_array == 1)
    if interior_coords.size == 0:
        return None
    centroid = tuple(np.round(np.mean(interior_coords, axis=0)).astype(int))
    if binary_image_array[centroid] == 1:
        return centroid
    return _center_pole(binary_image_array, interior_coords)

   