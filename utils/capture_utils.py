"""
capture_utils.py

Shared utilities for saving and organizing image data collected during the
B-series notebooks (image capture, dataset building).

NOTE ON SCOPE: This module intentionally does NOT manage the camera resource.
Camera access is provided by a background virtual-camera service that starts
at boot and owns the physical camera; TraitletCamera() may be instantiated
directly in any notebook and cannot block or conflict with another instance.
This module covers a different, still re-run-sensitive pattern: writing a
captured frame to a labeled dataset directory with a unique filename, and
keeping any on-screen file counts accurate.
"""

import os
from uuid import uuid1


def ensure_directory(path):
    """Creates the directory at `path` if it doesn't already exist.

    Safe to call on every notebook run -- re-running a cell will not raise
    an error or attempt to recreate an existing directory.

    Args:
        path (str): directory path to create

    Returns:
        str: the same path, so this can be chained, e.g.
             red_dir = ensure_directory("Datasets/red_blue/red/")
    """
    os.makedirs(path, exist_ok=True)
    return path


def generate_image_path(directory, label=None):
    """Builds a unique file path for a new image inside `directory`.

    Uses uuid1() so that repeated calls -- even across separate runs of the
    same notebook -- never collide with an existing filename.

    Args:
        directory (str): directory the image will be saved into
        label (str, optional): short prefix describing the image,
            e.g. "red", "blue". Omit for an unlabeled/generic image.

    Returns:
        str: e.g. "Datasets/red_blue/red/red_1a2b3c4d....jpg"
    """
    unique_id = uuid1()
    filename = f'{label}_{unique_id}.jpg' if label else f'{unique_id}.jpg'
    return os.path.join(directory, filename)


def save_image(image_bytes, directory, label=None):
    """Saves already-encoded image bytes to a uniquely-named file inside
    `directory`, creating the directory first if needed.

    `image_bytes` is expected to already be JPEG-encoded -- for example,
    an ipywidgets Image widget's `.value` after a `bgr8_to_jpeg` transform.
    This function does not perform any image encoding itself.

    Args:
        image_bytes (bytes): JPEG-encoded image data
        directory (str): destination directory for the image
        label (str, optional): prefix for the filename, e.g. a class label

    Returns:
        str: the full path the image was saved to
    """
    ensure_directory(directory)
    image_path = generate_image_path(directory, label)

    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    return image_path


def count_images(directory):
    """Counts the number of files currently in `directory`.

    Returns 0 if the directory doesn't exist yet rather than raising, which
    makes this safe to use for initializing a UI counter before any images
    have been saved, or after a fresh git reset on the fleet.

    Args:
        directory (str): directory to count files in

    Returns:
        int: number of files in the directory (0 if it doesn't exist)
    """
    if not os.path.exists(directory):
        return 0
    return len(os.listdir(directory))
