"""Render a separate README chart for cached continuation turns, from saved usage."""

# SVG layout literals keep their source lines intact.
# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = json.loads((ROOT / "experiments/summary/warm-cache.json").read_text())
    cards = []
    for index, (model, name) in enumerate(
        [("glm-5.3", "GLM-5.3"), ("claude-opus-4-8", "Claude Opus 4.8")]
    ):
        methods = report["models"][model]["continuation"]
        full, jev = methods["full"], methods["jev"]
        saving = 100 * (1 - jev["total_usd"] / full["total_usd"])
        rows = []
        for row, (method, label, color) in enumerate(
            [
                ("full", "Cached full context", "#7a8c83"),
                ("jev", "JEV selection", "#147d65"),
                ("summarization", "Reused summary (lowest cost)", "#b08039"),
            ]
        ):
            value = methods[method]["total_usd"]
            y = 246 + row * 77
            rows.append(f'''
    <text x="30" y="{y}" class="label">{label}</text>
    <text x="520" y="{y}" class="amount" text-anchor="end">${value:.4f}</text>
    <rect x="30" y="{y + 14}" width="490" height="25" rx="3" fill="#eef2ea"/>
    <rect x="30" y="{y + 14}" width="{490 * value / full["total_usd"]:.3f}" height="25" rx="3" fill="{color}"/>''')
        cards.append(f"""
  <g transform="translate({40 + 570 * index},210)">
    <rect width="550" height="502" rx="20" fill="#fff" stroke="#dcded5"/>
    <text x="28" y="44" class="model">{name}</text>
    <text x="25" y="134" class="saving">{saving:.1f}%</text>
    <text x="30" y="168" class="body">lower JEV cost vs. cached full context</text>
    <text x="30" y="200" class="detail">Full-context input cache hit: {100 * full["answer_cache_hit_fraction"]:.2f}%</text>
    {"".join(rows)}
    <line x1="30" y1="455" x2="520" y2="455" stroke="#e2e5dc"/>
    <text x="30" y="484" class="pass">All 3 methods: 8/8 passing artifacts</text>
  </g>""")
    svg = (
        """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="938" viewBox="0 0 1200 938" role="img" aria-labelledby="title desc">
  <title id="title">Cached continuation costs: full context, JEV, and reused summaries</title>
  <desc id="desc">Turns 2 and 3 only, summed over four histories. JEV costs less than cached full context, while reused summaries cost least. All three methods pass every artifact. Initial summary generation and initial full-history cache population are excluded from all methods.</desc>
  <style>
    text{font-family:Arial,Helvetica,sans-serif;fill:#163e32}
    .eyebrow{font-size:17px;font-weight:700;letter-spacing:2px}
    .headline{font-size:44px;font-weight:700;letter-spacing:-1.3px}
    .body{font-size:21px}.model{font-size:25px;font-weight:700}
    .saving{font-size:82px;font-weight:700;letter-spacing:-4px}
    .label{font-size:20px}.amount{font-size:23px;font-weight:700}
    .pass{font-size:22px;font-weight:700}.detail{font-size:18px;fill:#64746a}
    .note{font-size:18px;fill:#53655b}
  </style>
  <rect width="1200" height="938" rx="24" fill="#f6f5ed"/>
  <rect x="40" y="37" width="9" height="20" rx="2" fill="#b08039"/>
  <text x="62" y="54" class="eyebrow">TURNS 2–3 ONLY / INITIAL COSTS EXCLUDED</text>
  <text x="40" y="120" class="headline">What if the history is already cached?</text>
  <text x="40" y="163" class="body">The same code-generation experiment, viewed after the first turn.</text>
"""
        + "".join(cards)
        + """
  <rect x="40" y="735" width="1120" height="76" rx="12" fill="#f0e4cf"/>
  <text x="64" y="768" style="font-size:23px;font-weight:700;fill:#674917">Reused summaries were cheapest on these continuation turns.</text>
  <text x="64" y="794" style="font-size:18px;fill:#674917">JEV saved against cached full context; it did not beat the already-created summary here.</text>
  <text x="40" y="850" class="note">4 histories × 2 follow-up turns = 8 artifacts per model / method; 32 tests per artifact.</text>
  <text x="40" y="880" class="note">USD includes cache reads/writes, output and JEV fees for these turns. First-turn costs excluded.</text>
  <text x="40" y="910" class="note">Zero-baseline bars, scaled per model. This two-turn slice does not establish long-session savings.</text>
</svg>
"""
    )
    output = ROOT / "docs/assets/warm-cache-cost.svg"
    output.write_text(svg)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
