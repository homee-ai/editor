#!/usr/bin/env python3
"""Small VLM client for sending rendered SVG floor plans as images."""

from __future__ import annotations

import argparse
import base64
import json
import os
import http.client
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TRANSIENT_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.0-flash"
REASONING_EFFORT_VALUES = {"minimal", "low", "medium", "high"}


def expand_env_value(value: str) -> str:
    value = re.sub(r"^~(?=$|/|\\)", str(Path.home()), value)

    def replace_var(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, "")

    return re.sub(r"\$(\w+)|%(\w+)%", replace_var, value)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        values[key.strip()] = expand_env_value(value)
    return values


def load_editor_env(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    env = dict(os.environ)
    for path in [
        repo_root / ".env.local",
        repo_root / "apps/editor/.env.local",
        repo_root / ".env",
        repo_root / "apps/editor/.env",
    ]:
        for key, value in parse_env_file(path).items():
            env.setdefault(key, value)
    return env


def resolve_api_key(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith("file:"):
        try:
            return Path(expand_env_value(value[5:].strip())).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return value


def normalize_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in REASONING_EFFORT_VALUES:
        raise ValueError("Invalid reasoning effort. Use one of: minimal, low, medium, high.")
    return normalized


def render_svg_to_png(svg_path: Path, size: int) -> bytes:
    try:
        import cairosvg  # type: ignore

        return cairosvg.svg2png(url=str(svg_path), output_width=size)
    except Exception:
        return render_svg_with_qlmanage(svg_path, size)


def render_svg_with_qlmanage(svg_path: Path, size: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="floor2ifc-vlm-") as tmp:
        tmp_dir = Path(tmp)
        result = subprocess.run(
            ["qlmanage", "-t", "-s", str(size), "-o", str(tmp_dir), str(svg_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"qlmanage failed:\n{result.stdout}")

        pngs = sorted(tmp_dir.glob("*.png"))
        if not pngs:
            raise RuntimeError(f"qlmanage produced no PNG:\n{result.stdout}")
        return pngs[0].read_bytes()


def image_data_url(svg_path: Path, size: int = 1600) -> str:
    png_bytes = render_svg_to_png(svg_path, size)
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(prompt: str, image_urls: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    return [{"role": "user", "content": content}]


def extract_reply(data: Any) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in VLM response.")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("VLM response JSON is not an object.")
    return parsed


class VlmClient:
    def __init__(self, env_root: Path = REPO_ROOT) -> None:
        self.env_root = env_root.resolve()
        self.env = load_editor_env(self.env_root)
        self.provider = self.env.get("EDITOR_LLM_PROVIDER", "gemini")
        self.base_url = self.env.get("EDITOR_LLM_BASE_URL") or DEFAULT_BASE_URL
        self.model = self.env.get("EDITOR_LLM_MODEL") or DEFAULT_MODEL
        self.api_key = resolve_api_key(
            self.env.get("EDITOR_LLM_API_KEY")
            or self.env.get("GEMINI_API_KEY")
            or self.env.get("GOOGLE_API_KEY")
            or ""
        )
        self.temperature = float(self.env.get("EDITOR_LLM_TEMPERATURE", "0.2"))
        self.max_tokens = max(int(self.env.get("EDITOR_LLM_MAX_TOKENS", "4096")), 4096)
        self.reasoning_effort = normalize_reasoning_effort(
            self.env.get("EDITOR_LLM_REASONING_EFFORT")
        )

    def chat(self, prompt: str, image_urls: list[str], dump_request: bool = False) -> str:
        if not self.api_key and not re.search(
            r"^https?://(localhost|127\.0\.0\.1|\[::1\]|::1)", self.base_url
        ):
            raise RuntimeError(
                "Missing API key. Set EDITOR_LLM_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY."
            )

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(prompt, image_urls),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        if dump_request:
            print(
                json.dumps(
                    {
                        "provider": self.provider,
                        "base_url": self.base_url,
                        "url": url,
                        "model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "image_count": len(image_urls),
                        "image_data_url_bytes": [len(url) for url in image_urls],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        for attempt in range(MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    response_text = response.read().decode("utf-8", errors="replace")
                    status = response.status
            except urllib.error.HTTPError as error:
                status = error.code
                response_text = error.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
                # Network/timeout/connection drop: retry with backoff, raise after the last attempt.
                if attempt < MAX_RETRIES:
                    print(f"LLM request error ({error!r}); retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"LLM request failed: {error}") from error

            if status in TRANSIENT_STATUS and attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"LLM HTTP {status}; retry {attempt + 1}/{MAX_RETRIES} in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            break

        try:
            data: Any = json.loads(response_text)
        except ValueError:
            data = response_text

        if status < 200 or status >= 300:
            detail = json.dumps(data, ensure_ascii=False, indent=2) if not isinstance(data, str) else data
            raise RuntimeError(f"LLM request failed: HTTP {status}\n{detail}")

        reply = extract_reply(data)
        if not reply:
            detail = json.dumps(data, ensure_ascii=False, indent=2) if not isinstance(data, str) else data
            raise RuntimeError(f"Could not parse model reply.\n{detail}")
        return reply

    def chat_with_svgs(
        self,
        prompt: str,
        svg_paths: list[Path],
        image_size: int = 1600,
        dump_request: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        if max_tokens is not None:
            self.max_tokens = max(self.max_tokens, max_tokens)
        image_urls = [image_data_url(path, image_size) for path in svg_paths]
        return self.chat(prompt, image_urls, dump_request)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path, nargs="+", help="SVG image(s) to render and send.")
    parser.add_argument("--prompt", type=Path, required=True, help="Prompt text file.")
    parser.add_argument("--env-file-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--image-size", type=int, default=1600)
    parser.add_argument("--dump-request", action="store_true")
    args = parser.parse_args()

    for svg_path in args.svg:
        if not svg_path.exists():
            print(f"SVG not found: {svg_path}", file=sys.stderr)
            return 2
    if not args.prompt.exists():
        print(f"Prompt not found: {args.prompt}", file=sys.stderr)
        return 2

    prompt = args.prompt.read_text(encoding="utf-8")
    client = VlmClient(args.env_file_root)
    try:
        print(client.chat_with_svgs(prompt, args.svg, args.image_size, args.dump_request))
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
