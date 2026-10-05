#!/usr/bin/env python3
"""Package the VGG16-BN report as a small standalone static website."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "vgg16bn_share_site"
SITE = OUT / "dist"


def render_report(source: str, title: str) -> str:
    body = (PAPER / source).read_text(encoding="utf-8")
    body = body.replace("(vgg16bn_results_report.md)", "(full-report.html)")
    body = body.replace("(vgg16bn_training_curves.md)", "(training-curves.html)")

    # Source files elsewhere in the research repository are not part of this
    # reader-facing package. Keep the cited file names as plain text.
    def fix_link(match: re.Match[str]) -> str:
        label, target = match.groups()
        path = target.split("#", 1)[0]
        if path.startswith("../") or (path and not (SITE / path).exists()):
            return label
        return match.group(0)

    body = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", fix_link, body)
    content = MarkdownIt("default", {"html": True}).enable("table").render(body)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#233041;background:#f4f6f9}}
body{{margin:0}}main{{max-width:1080px;margin:0 auto;padding:28px 24px 70px;background:white;min-height:100vh}}
nav{{position:sticky;top:0;background:#fff;padding:14px 0;border-bottom:1px solid #dce3eb;z-index:2}}
nav a{{margin-right:24px;color:#00679b}}a{{color:#00679b}}h1,h2,h3{{line-height:1.25}}
h2{{margin-top:2.4rem;border-top:1px solid #dce3eb;padding-top:1.4rem}}
p,li{{line-height:1.55}}img{{display:block;max-width:100%;height:auto;margin:22px auto}}
table{{border-collapse:collapse;min-width:720px}}th,td{{border:1px solid #dce3eb;padding:7px 9px;text-align:left}}
th{{background:#edf2f7}}table{{display:block;overflow:auto;max-width:100%}}
details{{border:1px solid #dce3eb;border-radius:8px;margin:15px 0;padding:10px 14px}}
summary{{cursor:pointer;font-weight:650}}code{{background:#edf2f7;padding:1px 3px;border-radius:3px}}
</style></head><body><main><nav><a href="index.html">Interactive comparison</a><a href="full-report.html">Full report</a><a href="training-curves.html">Training curves</a></nav>{content}</main></body></html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    for name in ("vgg16bn_comparison", "vgg16bn_training"):
        shutil.copytree(PAPER / "figures" / name, SITE / "figures" / name, dirs_exist_ok=True)
    shutil.copytree(PAPER / "vgg16bn_report_data", SITE / "vgg16bn_report_data", dirs_exist_ok=True)

    html = (PAPER / "vgg16bn_results_report.html").read_text(encoding="utf-8")
    html = html.replace('href="vgg16bn_results_report.md"', 'href="full-report.html"')
    html = html.replace('href="vgg16bn_training_curves.md"', 'href="training-curves.html"')
    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / "full-report.html").write_text(
        render_report("vgg16bn_results_report.md", "VGG16-BN behavioral results"), encoding="utf-8"
    )
    (SITE / "training-curves.html").write_text(
        render_report("vgg16bn_training_curves.md", "VGG16-BN training curves"), encoding="utf-8"
    )
    (OUT / ".openai").mkdir(exist_ok=True)
    hosting = OUT / ".openai" / "hosting.json"
    if not hosting.exists():
        hosting.write_text('{"static":{"directory":"dist"}}\n', encoding="utf-8")
    print(f"Packaged static site at {OUT}")


if __name__ == "__main__":
    main()
