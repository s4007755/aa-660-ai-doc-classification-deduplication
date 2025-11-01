from src.utils.hash_utils import HashUtils, hash_text, hash_file_path, generate_deterministic_seed, create_label_hash

def test_hash_text_basic():
    h1 = hash_text("Hello World")
    h2 = hash_text(" hello world ")
    assert h1 == h2  # normalized
    assert len(h1) == 64

def test_hash_bytes_and_path(tmp_path):
    data = b"abc123"
    hb = HashUtils.hash_bytes(data)
    p = tmp_path / "f.txt"
    p.write_bytes(data)
    hf = hash_file_path(str(p))
    assert hb == hf

def test_deterministic_seed_and_label_hash():
    s1 = generate_deterministic_seed("foo")
    s2 = generate_deterministic_seed("foo")
    assert s1 == s2
    lh = create_label_hash("0", "Politics")
    assert lh.startswith("label_")
    assert HashUtils.verify_hash_consistency(" Politics ", HashUtils.hash_text("Politics"))

