"""Serve a blinded human review of old/new multiplicity disagreements."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel


LABELS = {
    "single": "单细胞",
    "touching_doublet": "粘连双细胞",
    "cluster_3plus": "3+细胞团",
    "debris": "杂质",
    "invalid": "无效（孔壁）",
}
VALID_LABELS = set(LABELS)


class ReviewItem(BaseModel):
    case_id: str
    label: str


class ReviewPayload(BaseModel):
    items: list[ReviewItem]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _render_crop(row: pd.Series, output: Path, *, contour: bool) -> None:
    crop_size = 112
    scale = 4
    with Image.open(str(row.raw_image_path)) as source:
        raw = np.asarray(source.convert("L"), dtype=np.uint8)
    x = float(row.x_px)
    y = float(row.y_px)
    left = int(round(x)) - crop_size // 2
    top = int(round(y)) - crop_size // 2
    canvas = np.full((crop_size, crop_size), int(np.median(raw)), dtype=np.uint8)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(raw.shape[1], left + crop_size), min(raw.shape[0], top + crop_size)
    canvas[sy0 - top : sy1 - top, sx0 - left : sx1 - left] = raw[sy0:sy1, sx0:sx1]
    image = ImageOps.autocontrast(Image.fromarray(canvas, mode="L"), cutoff=0.5).convert("RGB")
    image = image.resize((crop_size * scale, crop_size * scale), Image.Resampling.NEAREST)
    if contour:
        draw = ImageDraw.Draw(image)
        points = []
        value = row.get("v2_contour_json", "")
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                points = [
                    ((float(px) - left) * scale, (float(py) - top) * scale)
                    for px, py in json.loads(value)
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                points = []
        if len(points) > 2:
            draw.line(
                points + [points[0]], fill=(126, 238, 216), width=3, joint="curve"
            )
        else:
            radius = max(7.0, float(row.get("diameter_px", 10.0)) * 0.7) * scale
            cx, cy = (x - left) * scale, (y - top) * scale
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline=(126, 238, 216),
                width=3,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _load_cases(source_root: Path, output_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    frames = []
    for slug in ("ql2603-t3-1", "ql2603-t4-3"):
        path = source_root / slug / "unreviewed_single_doublet_disagreements.csv"
        frame = pd.read_csv(path, low_memory=False)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=20260820).reset_index(drop=True)
    cases: list[dict[str, Any]] = []
    secret: dict[str, dict[str, str]] = {}
    image_root = output_root / "images"
    for number, (_, row) in enumerate(combined.iterrows(), start=1):
        case_id = f"case-{number:03d}"
        plain_name = f"{case_id}-plain.png"
        contour_name = f"{case_id}-contour.png"
        _render_crop(row, image_root / plain_name, contour=False)
        _render_crop(row, image_root / contour_name, contour=True)
        cases.append(
            {
                "case_id": case_id,
                "number": number,
                "plate": str(row.plate).replace("ql2603-", "").upper(),
                "well": str(row.well),
                "timepoint": str(row.timepoint),
                "image_plain": f"/images/{plain_name}",
                "image_contour": f"/images/{contour_name}",
            }
        )
        secret[case_id] = {
            "plate": str(row.plate),
            "candidate_id": str(row.candidate_id),
            "old": str(row.old_predicted_multiplicity),
            "new": str(row.new_predicted_multiplicity),
        }
    return cases, secret


def _page() -> str:
    labels = json.dumps(LABELS, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QL2603 单双细胞盲审</title>
<style>
body{{margin:0;background:#edf3f0;color:#153f42;font-family:'Microsoft YaHei',sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#0d4e52;color:white;padding:16px 24px;box-shadow:0 2px 8px #0003}}
header h1{{font-size:24px;margin:0 0 6px}} header p{{margin:0;color:#cfe9e6}}
.bar{{display:flex;gap:14px;align-items:center;margin-top:12px}} progress{{width:260px;height:12px}}
main{{padding:20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:18px}}
.card{{background:white;border:2px solid transparent;border-radius:14px;overflow:hidden;box-shadow:0 3px 14px #20433a1a}}
.card.done{{border-color:#36a985}} .meta{{display:flex;justify-content:space-between;padding:12px 14px;font-weight:700}}
.image-wrap{{background:#dce6e2;text-align:center;position:relative}} .image-wrap img{{width:100%;max-width:448px;display:block;margin:auto;image-rendering:pixelated}}
.toggle{{position:absolute;right:9px;top:9px;border:1px solid #9cb6af;background:#ffffffdf;border-radius:8px;padding:6px 10px;cursor:pointer}}
.choices{{padding:12px;display:grid;grid-template-columns:repeat(2,1fr);gap:8px}}
.choice{{border:1px solid #a8bbb6;background:#f7faf9;border-radius:9px;padding:10px 5px;cursor:pointer;color:#244c4b}}
.choice:hover{{background:#e8f5f0}} .choice.selected{{background:#2d9d7d;color:white;border-color:#2d9d7d;font-weight:700}}
#submit{{border:0;border-radius:9px;padding:10px 18px;background:#e9ad38;color:#173a3c;font-weight:700;cursor:pointer}}
#submit:disabled{{opacity:.45;cursor:not-allowed}} #result{{font-weight:700;color:#ffe7aa}}
</style></head><body>
<header><h1>QL2603 T3-1 / T4-3 单双细胞盲审</h1>
<p>只看图像选择真实类别；页面不会显示哪项来自新模型或旧模型。浅青轮廓可逐图关闭。</p>
<div class="bar"><span id="count">0/0 已判定</span><progress id="progress" value="0" max="1"></progress><button id="submit" disabled>全部提交并解盲</button><span id="result"></span></div></header>
<main id="cards"></main>
<script>
const LABELS={labels}; let cases=[]; const answers={{}};
async function load(){{const r=await fetch('/api/cases');const d=await r.json();cases=d.cases;Object.assign(answers,d.answers);render();}}
function update(){{const n=Object.keys(answers).length;document.getElementById('count').textContent=`${{n}}/${{cases.length}} 已判定`;document.getElementById('progress').max=cases.length;document.getElementById('progress').value=n;document.getElementById('submit').disabled=n!==cases.length;}}
function render(){{const root=document.getElementById('cards');root.innerHTML='';for(const c of cases){{const card=document.createElement('section');card.className='card'+(answers[c.case_id]?' done':'');card.id=c.case_id;card.innerHTML=`<div class="meta"><span>#${{c.number}}　${{c.plate}}　${{c.well}}/${{c.timepoint}}</span><span class="state">${{answers[c.case_id]?LABELS[answers[c.case_id]]:'待判定'}}</span></div><div class="image-wrap"><img src="${{c.image_contour}}" data-plain="${{c.image_plain}}" data-contour="${{c.image_contour}}"><button class="toggle">隐藏轮廓</button></div><div class="choices"></div>`;const img=card.querySelector('img'),toggle=card.querySelector('.toggle');toggle.onclick=()=>{{const showing=img.src.includes('-contour.png');img.src=showing?img.dataset.plain:img.dataset.contour;toggle.textContent=showing?'显示轮廓':'隐藏轮廓';}};const choices=card.querySelector('.choices');for(const [value,label] of Object.entries(LABELS)){{const b=document.createElement('button');b.className='choice'+(answers[c.case_id]===value?' selected':'');b.textContent=label;b.onclick=()=>{{answers[c.case_id]=value;render();document.getElementById(c.case_id).scrollIntoView({{block:'center'}});}};choices.appendChild(b);}}root.appendChild(card);}}update();}}
document.getElementById('submit').onclick=async()=>{{const items=cases.map(c=>({{case_id:c.case_id,label:answers[c.case_id]}}));const r=await fetch('/api/reviews',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{items}})}});const d=await r.json();document.getElementById('result').textContent=d.message;if(d.complete) document.getElementById('submit').disabled=true;}};load();
</script></body></html>"""


def create_app(source_root: Path, output_root: Path) -> FastAPI:
    output_root.mkdir(parents=True, exist_ok=True)
    cases, secret = _load_cases(source_root, output_root)
    (output_root / "blind_key.json").write_text(
        json.dumps(secret, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reviews_path = output_root / "reviews.json"
    lock = Lock()

    def read_reviews() -> dict[str, str]:
        if not reviews_path.exists():
            return {}
        try:
            value = json.loads(reviews_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    app = FastAPI(title="QL2603 multiplicity blind review")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _page()

    @app.get("/api/ready")
    def ready() -> dict[str, Any]:
        return {"status": "ready", "case_count": len(cases), "reviewed": len(read_reviews())}

    @app.get("/api/cases")
    def api_cases() -> dict[str, Any]:
        return {"cases": cases, "answers": read_reviews()}

    @app.get("/images/{name}")
    def image(name: str) -> FileResponse:
        if Path(name).name != name:
            raise HTTPException(404)
        path = output_root / "images" / name
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/api/reviews")
    def save_reviews(payload: ReviewPayload) -> dict[str, Any]:
        valid_ids = set(secret)
        incoming = {item.case_id: item.label for item in payload.items}
        if set(incoming) != valid_ids or any(label not in VALID_LABELS for label in incoming.values()):
            raise HTTPException(400, "All blind cases require one valid label")
        with lock:
            temporary = reviews_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(incoming, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(reviews_path)
            rows = []
            old_correct = 0
            new_correct = 0
            for case in cases:
                case_id = case["case_id"]
                truth = incoming[case_id]
                hidden = secret[case_id]
                old_hit = hidden["old"] == truth
                new_hit = hidden["new"] == truth
                old_correct += int(old_hit)
                new_correct += int(new_hit)
                rows.append(
                    {
                        **case,
                        **hidden,
                        "reviewed_label": truth,
                        "old_correct": old_hit,
                        "new_correct": new_hit,
                    }
                )
            pd.DataFrame(rows).to_csv(
                output_root / "blind_review_results.csv", index=False, encoding="utf-8"
            )
            result = {
                "complete": True,
                "reviewed": len(cases),
                "old_correct": old_correct,
                "new_correct": new_correct,
                "old_accuracy": old_correct / len(cases),
                "new_accuracy": new_correct / len(cases),
            }
            (output_root / "blind_review_results.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        winner = "新模型" if new_correct > old_correct else "旧模型" if old_correct > new_correct else "两者相同"
        return {
            **result,
            "message": f"解盲完成：旧模型 {old_correct}/{len(cases)}，新模型 {new_correct}/{len(cases)}；{winner}更高。",
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()
    app = create_app(
        Path(args.source_root).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
