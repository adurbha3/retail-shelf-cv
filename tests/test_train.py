from retailshelf.synthetic import generate_dataset
from retailshelf.data import batch_tensors
from retailshelf.train import TrainConfig, train_detector, train_ocr
from retailshelf.ocr import generate_ocr_dataset, predict_digits


def test_train_detector_runs_and_produces_finite_weights():
    shelves = generate_dataset(6, seed=0)
    images, targets = batch_tensors(shelves)
    model = train_detector(images, targets, TrainConfig(epochs=1, batch_size=2))
    for param in model.parameters():
        assert bool((param == param).all())  # no NaNs


def test_train_ocr_runs_and_predicts():
    images, labels, _ = generate_ocr_dataset(8, seed=0)
    model = train_ocr(images, labels, TrainConfig(epochs=1, batch_size=4))
    predictions = predict_digits(model, images)
    assert len(predictions) == 8
