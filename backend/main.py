import os
import io
import httpx
import spacy
from spacy.matcher import PhraseMatcher
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── spaCy setup ───────────────────────────────────────────────────────────────
# Load small English model — only needs tokenizer + POS tagger, no NER
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ── Filler word list ──────────────────────────────────────────────────────────
# PhraseMatcher works on tokens so punctuation is already split off.
# Add/remove words here to tune for your accent.
FILLER_WORDS = [
    # English core
    "um", "uh", "ah", "hmm", "hm", "err", "umm", "uhh", "aah", "ooh",
    # Multi-word
    "you know", "i mean", "sort of", "kind of",
    # Common
    "like", "actually", "basically", "right", "well", "okay",
    "literally", "honestly", "clearly", "so",
    # Tamil / Indian English
    "na", "da", "yaar", "means", "only", "itself",
    "enna", "seri", "dei", "pa", "po",
]

# Context rules — when a word is NOT a filler even if it matches the list.
# Key = filler word, value = list of (preceding_pos, following_pos) pairs
# that indicate real usage. Uses spaCy POS tags.
NON_FILLER_CONTEXT = {
    "like": [
        # "feels like", "looks like" — verb before "like" = not a filler
        ("VERB", None),
        # "like this", "like that" — followed by DET = not a filler
        (None, "DET"),
    ],
    "well": [
        # "played well", "works well" — adverb after verb = not a filler
        ("VERB", None),
    ],
    "right": [
        # "right now", "right here" — followed by adverb = not a filler
        (None, "ADV"),
    ],
    "so": [
        # "so that" — followed by "that" = conjunction, not filler
        (None, "SCONJ"),
    ],
    "only": [
        # "only one", "only the" — followed by DET/NUM = not a filler
        (None, "DET"),
        (None, "NUM"),
    ],
}

# Build the PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(w) for w in FILLER_WORDS]
matcher.add("FILLER", patterns)

def is_filler_in_context(token, doc):
    """Return False if the token matches a NON_FILLER_CONTEXT rule."""
    word = token.lower_
    rules = NON_FILLER_CONTEXT.get(word, [])
    for prev_pos, next_pos in rules:
        prev_ok = (prev_pos is None) or (
            token.i > 0 and doc[token.i - 1].pos_ == prev_pos
        )
        next_ok = (next_pos is None) or (
            token.i < len(doc) - 1 and doc[token.i + 1].pos_ == next_pos
        )
        if prev_ok and next_ok:
            return False   # real usage, not a filler
    return True

def detect_fillers(text: str):
    doc = nlp(text)
    matches = matcher(doc)

    fillers = []
    counts = {}
    seen_spans = set()

    for match_id, start, end in matches:
        # Skip overlapping spans
        span_key = (start, end)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)

        span = doc[start:end]
        # Check context using the first token of the span
        if not is_filler_in_context(doc[start], doc):
            continue

        word = span.text.lower()
        counts[word] = counts.get(word, 0) + 1
        fillers.append({
            "text":       span.text,
            "start_char": span.start_char,
            "end_char":   span.end_char,
            "is_filler":  True,
        })

    # Build annotated segments for the frontend to highlight
    segments = []
    cursor = 0
    for f in sorted(fillers, key=lambda x: x["start_char"]):
        if f["start_char"] < cursor:
            continue
        if f["start_char"] > cursor:
            segments.append({
                "text":     text[cursor:f["start_char"]],
                "isFiller": False,
            })
        segments.append({
            "text":     text[f["start_char"]:f["end_char"]],
            "isFiller": True,
        })
        cursor = f["end_char"]
    if cursor < len(text):
        segments.append({"text": text[cursor:], "isFiller": False})

    return {"fillerCounts": counts, "segments": segments}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    api_key: str = Form(...),
):
    """
    Receive audio from the frontend, send to Sarvam, return transcript + fillers.
    Doing the Sarvam call server-side keeps the API key out of the browser.
    """
    audio_bytes = await file.read()

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": api_key},
            files={"file": ("chunk.webm", io.BytesIO(audio_bytes), "audio/webm")},
            data={
                "model": "saaras:v3",
                "mode":  "verbatim",
                "prompt": (
                    "Transcribe everything in English Roman script only. "
                    "Do not use Tamil, Hindi, or any Indic script. "
                    "Filler words like um, uh, ah, na, da, hmm should be "
                    "written in English letters."
                ),
            },
        )

    if r.status_code != 200:
        return {"error": r.text, "transcript": "", "fillerCounts": {}, "segments": []}

    data        = r.json()
    raw         = (data.get("transcript") or data.get("text") or "").strip()

    # Strip any Indic script that slips through despite the prompt
    import re
    text = re.sub(r"[\u0900-\u0DFF\u0E00-\u0FFF]+", "", raw).strip()

    result = detect_fillers(text)
    return {"transcript": text, **result}