"""Local VLM client (Qwen3-VL via Ollama) for pipeline reasoning.

Used for: scene hero naming (vision), M2 joint topology/type reasoning (text+geometry, optional
images), and render-critique loops. OpenAI-compatible /v1 endpoint, no API key, fully on-box.
"""
from __future__ import annotations
import os, json, base64, re, urllib.request

ENDPOINT = os.environ.get("ASSETFORGE_VLM", "http://localhost:11434/v1/chat/completions")
MODEL = os.environ.get("ASSETFORGE_VLM_MODEL", "qwen3-vl:32b-instruct")


def _b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def chat(prompt: str, images=None, temperature=0.0, max_tokens=512, model=None) -> str:
    content = [{"type": "text", "text": prompt}]
    for im in (images or []):
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _b64(im)}})
    payload = {"model": model or MODEL, "temperature": temperature,
               "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r["choices"][0]["message"]["content"]


def _json_block(text):
    m = re.search(r"\[.*\]|\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def available() -> bool:
    try:
        urllib.request.urlopen(ENDPOINT.replace("/chat/completions", "/models"), timeout=4)
        return True
    except Exception:
        return False


def name_heroes(image_path: str, max_heroes=6) -> list[str]:
    """Vision: list distinct foreground hero object labels in a scene image."""
    out = chat(f"List up to {max_heroes} distinct foreground HERO objects in this scene as a JSON "
               f"array of short singular noun labels, e.g. [\"tree\",\"rock\"]. Output only the array.",
               images=[image_path])
    j = _json_block(out)
    return [str(x) for x in j] if isinstance(j, list) else []


def joint_topology(parts: list[dict], images=None, legend=None) -> dict:
    """Reason VISUALLY about mechanical kinematics. parts = geometric descriptions; images =
    color-coded part renders (front/side/3q); legend = {part_name: [r,g,b]} mapping colors to parts.
    Returns {"base": name, "joints": [{"parent","child","type","motion"} ...]}.
    """
    leg = ""
    if legend:
        leg = "Each part is rendered in a distinct color:\n" + "\n".join(
            f"  {n}: RGB{tuple(c)}" for n, c in legend.items()) + "\n"
    prompt = (
        "You are a mechanical rigging expert. You are shown color-coded renders of an articulated "
        "object from several angles. " + leg +
        "Geometry (metric centroids, bbox sizes, contacts):\n" + json.dumps(parts, indent=2) +
        "\n\nLOOK at the images and infer how the object articulates. Return ONLY JSON: "
        "{\"base\": \"<static/grounded part>\", \"joints\": [{\"parent\":\"\",\"child\":\"\","
        "\"type\":\"fixed|revolute|prismatic|ball\",\"motion\":\"scan|elevate|open|spin|none\"} ...]}. "
        "Hints: a turret head spins about vertical (revolute, scan); a barrel/arm elevates (revolute, elevate); "
        "a door/drawer opens; the grounded part is the base. Build the chain base->...->tip.")
    out = chat(prompt, images=images, temperature=0.0, max_tokens=600)
    j = _json_block(out)
    return j if isinstance(j, dict) else {}


if __name__ == "__main__":
    import sys
    print("VLM available:", available(), "model:", MODEL)
    if len(sys.argv) > 1:
        print("heroes:", name_heroes(sys.argv[1]))
