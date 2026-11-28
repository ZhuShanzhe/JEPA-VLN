import base64, io, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERAS = ('right', 'center', 'left')

MODEL = "gpt-4o-2024-11-20"

PROMPT = """In an indoor object search, select ONE numbered detector proposal worth revisiting for the named goal, or null if none has a credible object-specific visual cue. Each row is one actual view from the same position. On the left is the unmarked image; on the right is the same image with possible candidate boxes and their IDs. The views are right, center and left. Compare ALL of them. Each box is only a suggestion and may mark the wrong object. A plausible room or support surface alone is insufficient. An identifiable target or a partially seen candidate with distinctive evidence is worth selecting. When several boxes represent the same plausible target, select the clearest one. Return exactly one JSON object, {"candidate":"ID"} or {"candidate":null}."""

PREFIXES = dict(right="R", center="C", left="L")

def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def to_png(views):
    size, head = 384, 27
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    dst = Image.new("RGB", (size * 2, (size + head) * 3))
    draw = ImageDraw.Draw(dst)
    for n, view in enumerate(CAMERAS):
        y = (size + head) * n
        data = views[view]
        with Image.open(data["path"]) as im:
            width, height = im.size
            fit = size / max(width, height)
            out_w,out_h = max(1,round(width*fit)),max(1,round(height*fit))
            dx,dy = (size-out_w)//2,(size-out_h)//2
            resized = im.convert("RGB").resize((out_w,out_h), Image.Resampling.NEAREST)
            rgb = Image.new("RGB",(size,size))
            rgb.paste(resized,(dx,dy))
        marked = rgb.copy()
        overlay = ImageDraw.Draw(marked)
        for i, (key, b) in enumerate(data["boxes"].items()):
            x1, y1, x2, y2 = [
                int(np.clip(offset + round(float(v)*factor), 0, size-1))
                for v,offset,factor in zip(b["xyxy"], (dx,dy,dx,dy), (out_w/width,out_h/height)*2)
            ]
            color = ["yellow", "cyan", "lime", "magenta"][i % 4]
            overlay.rectangle((x1, y1, x2, y2), outline=color, width=2)
            tx, ty = min(x1, size - 42), max(0, y1 - 21)
            overlay.rectangle((tx, ty, tx + 39, ty + 19), fill="black")
            overlay.text((tx + 1, ty), key, font=font, fill=color)
        draw.text((4, y+3), f"{view.upper()} | original", font=font)
        draw.text((size + 4, y+3), f"{view.upper()} | candidate IDs", font=font)
        dst.paste(rgb, (0, y + head))
        dst.paste(marked, (size, y + head))
    data = io.BytesIO()
    dst.save(data, format="PNG")
    return data.getvalue()

def materials(views):
    goals = {d["source"]["goal"] for d in views.values()}
    assert len(goals) == 1
    offered = [key for v in CAMERAS for key in views[v]["boxes"]]
    label = "Goal: " + goals.pop() + ". Offered IDs: " + (", ".join(offered) or "none") + ". Return JSON."
    sheet = to_png(views)
    request = dict(model=MODEL, temperature=0, max_tokens=40, response_format={"type":"json_object"},
                   messages=[{"role":"system","content":PROMPT},{"role":"user","content":[
                       {"type":"text","text":label},
                       {"type":"image_url","image_url":{"detail":"high","url":"data:image/png;base64," +
                                                                    base64.b64encode(sheet).decode()}}]}])
    return request, offered
