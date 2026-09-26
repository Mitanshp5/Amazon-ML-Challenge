from er.metrics import f05_single, macro_f05


def test_pdf_example():
    assert abs(f05_single({"S2-00047", "S3-00812"},
                          {"S2-00047", "S2-00193", "S3-00812"}) - 0.7142857) < 1e-6


def test_singletons():
    assert f05_single(set(), set()) == 1.0
    assert f05_single(set(), {"S2-1"}) == 0.0
    assert f05_single({"S2-1"}, set()) == 0.0


def test_macro_includes_singletons():
    truth = {"a": [], "b": ["S2-1"]}
    pred = {"a": [], "b": ["S2-1"]}
    assert macro_f05(truth, pred) == 1.0
