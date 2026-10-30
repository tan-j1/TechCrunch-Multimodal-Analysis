"""Score pitch-level inclusive collective language from a complete lead transcript."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "prompts" / "pitch_scores_only.txt"
MODEL = "gpt-4o-mini"
FIELDS = {"inclusive_collective_language"}


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_transcript(path):
    """Read full lead text, or select the lead's chronological turns in a blob."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        text = path.read_text(encoding="utf-8-sig").strip()
        video_id = path.stem
    else:
        record = read_json(path)
        if not isinstance(record, dict):
            raise ValueError("JSON input must be an object containing text or turns.")
        video_id = record.get("video_id", record.get("meta", {}).get("video_id", path.stem))
        text = record.get("text")
        if text is None:
            lead = record.get("lead", record.get("segmentation", {}).get("lead_speaker"))
            if not lead:
                raise ValueError("Turn-based input requires lead or segmentation.lead_speaker.")
            turns = record.get("turns", [])
            selected = [t for t in turns if t.get("speaker") == lead]
            selected.sort(key=lambda t: t.get("turn_id", t.get("start", 0)))
            text = "\n".join(t["text"].strip() for t in selected if t.get("text", "").strip())
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        text = text.strip()
    if not text:
        raise ValueError("The complete available lead-speaker transcript is empty.")
    return str(video_id), text


def build_payload(text, prompt):
    return {
        "model": MODEL, "temperature": 0.2, "max_tokens": 160,
        "n": 1, "stream": False, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in model response.")
        result[key] = value
    return result


def validate_response(body):
    """Validate the response model and score schema."""
    if not isinstance(body, dict):
        raise ValueError("Response must be an object.")
    returned_model = body.get("model")
    if not isinstance(returned_model, str) or not (returned_model == MODEL or returned_model.startswith(MODEL + "-")):
        raise ValueError("Response model differs from gpt-4o-mini.")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Response must contain exactly one choice.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Response choice must be an object.")
    message = choice.get("message", {})
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        raise ValueError("Incomplete model response.")
    if message.get("refusal") or message.get("tool_calls") or message.get("function_call"):
        raise ValueError("The model returned a refusal or tool call.")
    if not isinstance(message.get("content"), str):
        raise ValueError("Response content must be a JSON string.")
    scores = json.loads(message["content"], object_pairs_hook=unique_object)
    if not isinstance(scores, dict) or set(scores) != FIELDS:
        raise ValueError("Response score keys differ from the request schema.")
    if any(type(value) is not int or value not in (0, 1, 2) for value in scores.values()):
        raise ValueError("Each score must be an integer 0, 1, or 2.")
    return scores["inclusive_collective_language"]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def execute_once(payload):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Set OPENAI_API_KEY to the API credential for the selected endpoint.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not base_url.startswith("https://"):
        raise ValueError("OPENAI_BASE_URL must use HTTPS.")
    request = Request(
        base_url + "/chat/completions", method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with build_opener(NoRedirect()).open(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        # API error bodies can contain request data; report only HTTP status.
        raise ValueError(f"API request failed with HTTP {error.code}.") from None
    except URLError:
        raise ValueError("API endpoint could not be reached.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Complete lead-speaker transcript (.txt or .json).")
    parser.add_argument("--output", type=Path, required=True, help="Request payload or inclusive-language score JSON.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--response", type=Path, help="Validate a saved chat completion locally.")
    mode.add_argument("--execute", action="store_true", help="Send exactly one request using environment credentials.")
    args = parser.parse_args(argv)
    video_id, text = read_transcript(args.input)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    payload = build_payload(text, prompt)
    if args.response is None and not args.execute:
        write_json(args.output, payload)
        print(f"Prepared request: {args.output}")
        return 0
    body = read_json(args.response) if args.response else execute_once(payload)
    score = validate_response(body)
    result = {
        "video_id": video_id,
        "inclusive_collective_language": score,
        "text_sha256": sha256(text),
        "model": body["model"],
        "source": "saved_response" if args.response else "api_response",
    }
    if args.execute:
        result.update(prompt_sha256=sha256(prompt), temperature=payload["temperature"])
    write_json(args.output, result)
    if args.execute:
        write_json(args.output.with_name(args.output.stem + "_response.json"), body)
    print(f"Inclusive collective language = {score}; saved {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
