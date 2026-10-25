import torch

from retailshelf.synthetic import generate_shelf_image, generate_dataset, ROWS, COLS, IMG_WIDTH, IMG_HEIGHT
from retailshelf.data import image_to_tensor, shelf_to_target, batch_tensors


def test_image_to_tensor_shape_and_range():
    shelf = generate_shelf_image(seed=1)
    tensor = image_to_tensor(shelf)
    assert tensor.shape == (3, IMG_HEIGHT, IMG_WIDTH)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_shelf_to_target_marks_occupied_cells():
    shelf = generate_shelf_image(occupancy_rate=1.0, seed=2)
    target = shelf_to_target(shelf)
    assert target.shape == (ROWS, COLS, 5)
    assert torch.all(target[..., 0] == 1.0)


def test_shelf_to_target_empty_cells_have_zero_offset():
    shelf = generate_shelf_image(occupancy_rate=0.0, seed=3)
    target = shelf_to_target(shelf)
    assert torch.all(target[..., 0] == 0.0)
    assert torch.all(target[..., 1:5] == 0.0)


def test_batch_tensors_matches_dataset_size():
    shelves = generate_dataset(4, seed=0)
    images, targets = batch_tensors(shelves)
    assert images.shape == (4, 3, IMG_HEIGHT, IMG_WIDTH)
    assert targets.shape == (4, ROWS, COLS, 5)
