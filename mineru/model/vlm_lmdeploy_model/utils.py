import re

import torch


def select_best_resolution(original_size: tuple,
                           possible_resolutions: list) -> tuple:
    original_width, original_height = original_size
    best_fit = (0, 0)
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height,
                                   original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if isinstance(grid_pinpoints, str) and 'x' in grid_pinpoints:
        assert patch_size in [
            224, 336, 384, 448, 512
        ], 'patch_size should be in [224, 336, 384, 448, 512]'
        matches = re.findall(r'\((\d+)x(\d+)\)', grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [(i, j)
                          for i in range(range_start[0], range_end[0] + 1)
                          for j in range(range_start[1], range_end[1] + 1)]
        grid_pinpoints = [[dim * patch_size for dim in pair]
                          for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)  # type: ignore
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size
