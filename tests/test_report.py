from sleeplab.cli import main


def test_demo_builds_report(tmp_path):
    out = tmp_path / "reports"
    main(["--data", str(tmp_path / "data"), "demo", "--nights", "3", "--out", str(out)])
    pages = sorted(p.name for p in out.glob("*.html"))
    assert pages == ["index.html", "night-2026-10-23.html", "night-2026-10-24.html", "night-2026-10-25.html"]
    idx = (out / "index.html").read_text()
    assert "SYNTHETIC DATA" in idx and "not a medical device" in idx
    assert (out / "plotly.min.js").stat().st_size > 1_000_000  # offline: no CDN


def test_cover_renders(tmp_path):
    p = tmp_path / "cover.png"
    main(["cover", "--out", str(p)])
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
