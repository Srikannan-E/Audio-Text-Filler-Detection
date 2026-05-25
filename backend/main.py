import os
import io
import re
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
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ── Two tiers of filler words ─────────────────────────────────────────────────
#
# TIER 1 — ALWAYS fillers. These are pure hesitation sounds with no other
# meaning in English. Never run context checks on these — if spaCy sees them,
# they are fillers, period. This fixes the bug where "um I went to the store"
# was not highlighting "um" because the POS context check was too aggressive.
#
ALWAYS_FILLERS = {
    "um", "uh", "ah", "hmm", "hm", "err", "umm", "uhh",
    "aah", "ooh", "ohh",
    # Tamil / Indian English pure sounds
    "na", "da", "yaar", "enna", "seri", "dei", "pa", "po",
}

# TIER 2 — CONTEXTUAL fillers. Real English words that are also used as fillers.
# We use spaCy POS tags to decide: if the POS tag indicates a real grammatical
# role, skip it. If it looks like an interjection or discourse marker, flag it.
#
CONTEXTUAL_FILLERS = {
    "like", "actually", "basically", "right", "well", "okay",
    "literally", "honestly", "clearly", "so",
    "you know", "i mean", "sort of", "kind of",
    "means", "only", "itself",
}

# For contextual fillers: which POS tags indicate REAL usage (not a filler).
# If the token's POS is in this set, skip it.
NON_FILLER_POS = {
    "like":      {"VERB", "ADP"},          # "I like it", "looks like"
    "well":      {"ADV"},                   # "played well"
    "right":     {"ADJ", "ADV", "NOUN"},   # "right turn", "right now"
    "so":        {"SCONJ", "CCONJ"},       # "so that", "and so"
    "only":      {"ADV", "ADJ"},           # "only one", "the only way"
    "actually":  set(),                    # almost always a filler — no exclusions
    "basically": set(),
    "literally": set(),
    "honestly":  set(),
    "clearly":   set(),
    "okay":      set(),
    "means":     {"VERB"},                 # "it means" = real usage
    "itself":    {"PRON"},                 # "the thing itself" = real usage
}

ALL_FILLERS = ALWAYS_FILLERS | CONTEXTUAL_FILLERS

# ── Build PhraseMatcher ────────────────────────────────────────────────────────
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(w) for w in ALL_FILLERS]
matcher.add("FILLER", patterns)

# ── Detection logic ────────────────────────────────────────────────────────────
def is_filler(span_text: str, token, doc) -> bool:
    word = span_text.lower().strip()

    # Tier 1 — always a filler, no context check needed
    if word in ALWAYS_FILLERS:
        return True

    # Tier 2 — check POS tag against exclusion set
    exclusions = NON_FILLER_POS.get(word, set())
    if exclusions and token.pos_ in exclusions:
        return False   # real grammatical usage

    return True


def detect_fillers(text: str):
    doc = nlp(text)
    matches = matcher(doc)

    fillers = []
    counts  = {}
    seen    = set()

    for _, start, end in matches:
        if (start, end) in seen:
            continue
        seen.add((start, end))

        span  = doc[start:end]
        token = doc[start]

        if not is_filler(span.text, token, doc):
            continue

        word = span.text.lower()
        counts[word] = counts.get(word, 0) + 1
        fillers.append({
            "start_char": span.start_char,
            "end_char":   span.end_char,
        })

    # Build annotated segments
    segments = []
    cursor   = 0
    for f in sorted(fillers, key=lambda x: x["start_char"]):
        if f["start_char"] < cursor:
            continue
        if f["start_char"] > cursor:
            segments.append({"text": text[cursor:f["start_char"]], "isFiller": False})
        segments.append({"text": text[f["start_char"]:f["end_char"]], "isFiller": True})
        cursor = f["end_char"]
    if cursor < len(text):
        segments.append({"text": text[cursor:], "isFiller": False})

    return {"fillerCounts": counts, "segments": segments}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    api_key: str = Form(...),
):
    audio_bytes = await file.read()

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": api_key},
            files={"file": ("recording.webm", io.BytesIO(audio_bytes), "audio/webm")},
            data={
                "model": "saaras:v3",
                "mode":  "verbatim",
                "prompt": (
                    "Transcribe everything in English Roman script only. "
                    "Do not use Tamil, Hindi, or any Indic script. "
                    "Filler words like um, uh, ah, na, da, hmm must be "
                    "written in English letters exactly as spoken."
                ),
            },
        )

    if r.status_code != 200:
        return {
            "error": r.text,
            "transcript": "",
            "fillerCounts": {},
            "segments": [],
        }

    data = r.json()
    raw  = (data.get("transcript") or data.get("text") or "").strip()

    # Strip any Indic script that slips through
    text = re.sub(r"[\u0900-\u0DFF\u0E00-\u0FFF]+", "", raw).strip()

    print(f"[Sarvam verbatim] {repr(text)}")

    result = detect_fillers(text)
    return {"transcript": text, **result}