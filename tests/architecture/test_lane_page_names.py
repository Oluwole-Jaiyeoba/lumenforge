"""The four public lane names stay consistent across their entry points."""

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LANES = {
    "HARNESS_SIGNAL_BENCHMARK.html": "Harness Signal Benchmark",
    "CONTROLLER_POLICY_RESULTS.html": "Controller Policy Results",
    "GPU_INTERFERENCE.html": "GPU Interference",
    "KV_LIFECYCLE_AUDIT.html": "KV Lifecycle Audit",
}


class Headings(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current = ""
        self.title = ""
        self.h1 = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"title", "h1"}:
            self.current = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self.current:
            self.current = ""

    def handle_data(self, data: str) -> None:
        if self.current == "title":
            self.title += data
        elif self.current == "h1" and not self.h1:
            self.h1 = data


def test_lane_page_names_match_navigation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for filename, name in LANES.items():
        headings = Headings()
        headings.feed((ROOT / filename).read_text(encoding="utf-8"))
        assert headings.title == name
        assert headings.h1 == name
        assert f"[{name}]({filename})" in readme
