"""Render the README's standalone SVG from the recorded final-artifact results."""

# SVG layout literals keep their source lines intact.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = json.loads((ROOT / "experiments/summary/final-artifacts.json").read_text())
    cards = []
    for index, (model, label) in enumerate(
        [("glm-5.3", "GLM-5.3"), ("claude-opus-4-8", "Claude Opus 4.8")]
    ):
        methods = data["models"][model]["methods"]
        summary, jev = methods["summarization"], methods["jev"]
        ratio = jev["total_usd"] / summary["total_usd"]
        x = 40 + index * 570
        cards.append(f'''
  <g transform="translate({x},200)">
    <rect width="550" height="416" rx="20" fill="#ffffff" stroke="#dcded5"/>
    <text x="28" y="44" class="model">{label}</text>
    <text x="25" y="134" class="saving">{100 * (1 - ratio):.1f}%</text>
    <text x="30" y="166" class="body" fill="#35644e">lower cost than summarization</text>
    <text x="30" y="213" class="label">Summarization</text>
    <text x="520" y="213" class="amount" text-anchor="end">${summary["total_usd"]:.4f}</text>
    <rect x="30" y="227" width="490" height="28" rx="4" fill="#b9c5b2"/>
    <text x="30" y="292" class="label">JEV selection</text>
    <text x="520" y="292" class="amount" text-anchor="end">${jev["total_usd"]:.4f}</text>
    <rect x="30" y="306" width="490" height="28" rx="4" fill="#eef2ea"/>
    <rect x="30" y="306" width="{490 * ratio:.3f}" height="28" rx="4" fill="#147d65"/>
    <line x1="30" y1="358" x2="520" y2="358" stroke="#e2e5dc"/>
    <text x="30" y="389" class="pass">{summary["passed_artifacts"]}/{summary["artifacts"]} vs {jev["passed_artifacts"]}/{jev["artifacts"]} correct artifacts</text>
    <text x="520" y="389" class="detail" text-anchor="end">Summary / JEV</text>
  </g>''')
    svg = (
        """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="832" viewBox="0 0 1200 832" role="img" aria-labelledby="title description">
  <title id="title">JEV reduces measured final-code generation cost: GLM 83.9%, Opus 4.8 91.8%</title>
  <desc id="description">Both summarization and JEV produced 12 out of 12 passing Python artifacts per model. Every artifact passed 32 functional tests. GLM cost fell from $0.5087 to $0.0820; Opus 4.8 from $3.0379 to $0.2505. Bars start at zero and use each model's summarization cost as their own baseline. Costs include caching and JEV fees.</desc>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: #163e32; }
    .eyebrow { font-size: 17px; font-weight: 700; letter-spacing: 2px; }
    .headline { font-size: 48px; font-weight: 700; letter-spacing: -1.4px; }
    .body { font-size: 21px; }
    .model { font-size: 25px; font-weight: 700; }
    .saving { font-size: 82px; font-weight: 700; letter-spacing: -4px; }
    .label { font-size: 20px; }
    .amount { font-size: 23px; font-weight: 700; }
    .pass { font-size: 21px; font-weight: 700; }
    .detail { font-size: 16px; fill: #64746a; }
    .note { font-size: 18px; fill: #53655b; }
  </style>
  <rect width="1200" height="832" rx="24" fill="#f6f5ed"/>
  <rect x="40" y="37" width="9" height="20" rx="2" fill="#147d65"/>
  <text x="62" y="54" class="eyebrow">ALL 3 TURNS / INCLUDING INITIAL SUMMARY GENERATION</text>
  <text x="40" y="121" class="headline">Smaller context. Lower cost.</text>
  <text x="40" y="163" class="body">Total cost vs. summarization, including its first summary call. Final code tested.</text>
"""
        + "".join(cards)
        + """
  <rect x="40" y="639" width="1120" height="72" rx="12" fill="#163e32"/>
  <text x="64" y="670" style="fill:#e9f5dc;font-size:21px;font-weight:700">32 functional tests per artifact. Every test must pass.</text>
  <text x="64" y="695" style="fill:#d0dfd5;font-size:17px">Boundary values · invalid inputs · rejection priority · exact metadata and output types</text>
  <text x="40" y="746" class="note">4 histories × 3 turns per model / method · Usage-based USD, including cache and JEV fees.</text>
  <text x="40" y="776" class="note">Zero-baseline bars, normalized separately to each model’s summarization cost.</text>
  <text x="40" y="806" class="note">Task: request-admission policy generation. These results do not establish general coding quality.</text>
</svg>
"""
    )
    output = ROOT / "docs/assets/final-output-cost.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
