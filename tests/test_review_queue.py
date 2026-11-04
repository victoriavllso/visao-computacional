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
