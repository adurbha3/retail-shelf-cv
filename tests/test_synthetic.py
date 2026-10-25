from retailshelf.synthetic import generate_shelf_image, generate_dataset, ROWS, COLS, IMG_WIDTH, IMG_HEIGHT


def test_generates_correct_number_of_slots():
    shelf = generate_shelf_image(seed=1)
    assert len(shelf.slots) == ROWS * COLS


def test_image_has_expected_size():
    shelf = generate_shelf_image(seed=2)
    assert shelf.image.size == (IMG_WIDTH, IMG_HEIGHT)


def test_occupied_slots_have_price_sku_and_subregions():
    shelf = generate_shelf_image(occupancy_rate=1.0, seed=3)
    for slot in shelf.slots:
        assert slot.occupied
        assert slot.price is not None and 0.99 <= slot.price <= 29.99
        assert slot.sku is not None and 0 <= slot.sku <= 999999
        assert slot.price_tag_bbox is not None
        assert slot.barcode_bbox is not None


def test_empty_occupancy_rate_yields_no_products():
    shelf = generate_shelf_image(occupancy_rate=0.0, seed=4)
    assert all(not slot.occupied for slot in shelf.slots)
    assert all(slot.price is None for slot in shelf.slots)


def test_seeded_generation_is_reproducible():
    a = generate_shelf_image(seed=42)
    b = generate_shelf_image(seed=42)
    assert [s.occupied for s in a.slots] == [s.occupied for s in b.slots]
    assert list(a.image.getdata()) == list(b.image.getdata())


def test_generate_dataset_returns_requested_count():
    shelves = generate_dataset(5, seed=0)
    assert len(shelves) == 5
