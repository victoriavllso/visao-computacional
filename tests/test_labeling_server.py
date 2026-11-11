import time

from labeling.server import review_queue


def epoch(local_time):
    return time.mktime(time.strptime(local_time, "%Y-%m-%dT%H:%M:%S"))


def test_label_older_than_its_photo_is_reviewed():
    labels = {
        "123": {"status": "ok", "basis": "100g", "saved_at": "2026-10-11T10:28:04"},
        "456": {"status": "ok", "basis": "100g", "saved_at": "2026-10-20T09:00:00"},
    }
    # 123's photo was replaced after it was labeled; 456's was not.
    photo_times = {
        "123": epoch("2026-10-15T12:00:00"),
        "456": epoch("2026-10-15T12:00:00"),
    }

    reasons = review_queue(labels, photo_times)

    assert reasons == {"123": "photo replaced after it was labeled"}


def test_front_label_validation():
    from labeling.server import valid_front_label

    good = {
        "code": "789",
        "status": "ok",
        "added_sugars": True,
        "saturated_fat": False,
        "sodium": False,
        "boxes": [[10, 20, 300, 120]],
    }
    assert valid_front_label(good)
    assert valid_front_label({**good, "boxes": []})
    assert not valid_front_label({**good, "status": "maybe"})
    assert not valid_front_label({**good, "sodium": "yes"})
    assert not valid_front_label({**good, "boxes": [[10, 20, 300]]})
    assert not valid_front_label({**good, "boxes": [[-1, 20, 300, 120]]})


def test_replaced_photos_come_first():
    labels = {
        "111": {"status": "ok", "saved_at": "2026-10-20T09:00:00"},  # no basis
        "999": {"status": "ok", "basis": "100g", "saved_at": "2026-10-11T10:00:00"},
    }
    photo_times = {"999": epoch("2026-10-15T12:00:00")}

    assert list(review_queue(labels, photo_times)) == ["999", "111"]
