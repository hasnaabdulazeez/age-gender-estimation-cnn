from agegender.utkface import Sample, age_bucket, list_samples, parse_filename, split


def test_parse_valid_filename():
    assert parse_filename("26_1_0_20170116174525125.jpg.chip.jpg") == (26, 1)


def test_parse_rejects_malformed_names():
    assert parse_filename("39_1_20170116174525125.jpg") == (39, 1)  # missing race is OK
    assert parse_filename("61_3_20170109150557335.jpg") is None      # invalid gender
    assert parse_filename("abc_0_1_2017.jpg") is None
    assert parse_filename("25.jpg") is None


def test_list_samples_filters_by_age(tmp_path):
    for name in ["5_0_0_x.jpg", "30_1_2_x.jpg", "105_0_0_x.jpg", "notes.txt", "bad.jpg"]:
        (tmp_path / name).write_bytes(b"")
    samples = list_samples(tmp_path, min_age=1, max_age=100)
    assert sorted(s.age for s in samples) == [5, 30]


def test_split_is_deterministic_and_disjoint():
    samples = [Sample(f"{i}.jpg", 20 + i % 50, i % 2) for i in range(100)]
    a = split(samples, seed=1)
    b = split(samples, seed=1)
    assert a == b
    train, val, test = a
    assert (len(train), len(val), len(test)) == (80, 10, 10)
    assert not {s.path for s in train} & {s.path for s in test}


def test_age_bucket():
    assert age_bucket(8) == "0-12"
    assert age_bucket(35) == "30-39"
    assert age_bucket(77) == "60+"
