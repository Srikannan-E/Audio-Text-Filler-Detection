import os
import io
import re
import json
import httpx
import spacy
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# spaCy
# ─────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ─────────────────────────────────────────────────────────────
# Fillers
# ─────────────────────────────────────────────────────────────

ALWAYS_FILLERS = {
    "um", "uh", "ah", "hmm", "hm", "err", "umm", "uhh",
    "aah", "ooh", "ohh",

    # Tamil / Indian English
    "na", "da", "yaar", "enna", "seri", "dei", "pa", "po",
    "machi", "bro", "macha",
}

CONTEXTUAL_FILLERS = {
    "like", "actually", "basically", "right", "well", "okay",
    "literally", "honestly", "clearly", "so",
    "you know", "i mean", "sort of", "kind of",
    "means", "only", "itself",
}

NON_FILLER_POS = {
    "like":      {"VERB", "ADP"},
    "well":      {"ADV"},
    "right":     {"ADJ", "ADV", "NOUN"},
    "so":        {"SCONJ", "CCONJ"},
    "only":      {"ADV", "ADJ"},
    "actually":  set(),
    "basically": set(),
    "literally": set(),
    "honestly":  set(),
    "clearly":   set(),
    "okay":      set(),
    "means":     {"VERB"},
    "itself":    {"PRON"},
}

ALL_FILLERS = ALWAYS_FILLERS | CONTEXTUAL_FILLERS

_FILLER_PATTERNS = sorted(ALL_FILLERS, key=len, reverse=True)

_FILLER_RE = re.compile(
    r"(?<!\w)(" +
    "|".join(re.escape(p) for p in _FILLER_PATTERNS) +
    r")(?!\w)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────
# DEBUG LOGGER
# ─────────────────────────────────────────────────────────────

def debug_log(title, value):
    print("\n" + "=" * 80)
    print(f"🔹 {title}")
    print("-" * 80)
    print(value)
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────
# STEP 1 — PRESPLIT
# ─────────────────────────────────────────────────────────────

def _presplit(text: str):

    debug_log("STEP 1 - ORIGINAL INPUT", repr(text))

    chunks = []

    cursor = 0

    for m in _FILLER_RE.finditer(text):

        if m.start() > cursor:
            normal_text = text[cursor:m.start()]

            chunks.append({
                "text": normal_text,
                "start": cursor,
                "end": m.start(),
                "is_candidate": False,
            })

        filler_text = m.group()

        chunks.append({
            "text": filler_text,
            "start": m.start(),
            "end": m.end(),
            "is_candidate": True,
        })

        cursor = m.end()

    if cursor < len(text):
        chunks.append({
            "text": text[cursor:],
            "start": cursor,
            "end": len(text),
            "is_candidate": False,
        })

    debug_log("STEP 1 RESULT - CHUNKS", chunks)

    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 2 — VALIDATE CONTEXT
# ─────────────────────────────────────────────────────────────

def _validate_candidate(word_lower: str, left_context: str):

    debug_log(
        "STEP 2 - VALIDATING",
        {
            "candidate": word_lower,
            "left_context": left_context[-60:]
        }
    )

    # ALWAYS fillers
    if word_lower in ALWAYS_FILLERS:

        debug_log(
            "STEP 2 RESULT",
            f"{word_lower} => ALWAYS_FILLER => TRUE"
        )

        return True

    exclusions = NON_FILLER_POS.get(word_lower, set())

    if not exclusions:

        debug_log(
            "STEP 2 RESULT",
            f"{word_lower} => NO_EXCLUSION => TRUE"
        )

        return True

    context_text = (left_context[-60:] + " " + word_lower).strip()

    debug_log(
        "STEP 2 - SPACY INPUT",
        context_text
    )

    doc = nlp(context_text)

    tokens_debug = []

    for token in doc:
        tokens_debug.append({
            "text": token.text,
            "pos": token.pos_,
            "tag": token.tag_
        })

    debug_log("STEP 2 - SPACY TOKENS", tokens_debug)

    if not doc:

        debug_log(
            "STEP 2 RESULT",
            f"{word_lower} => EMPTY_DOC => TRUE"
        )

        return True

    candidate_token = doc[-1]

    debug_log(
        "STEP 2 - CANDIDATE POS",
        {
            "word": candidate_token.text,
            "pos": candidate_token.pos_,
            "excluded": list(exclusions)
        }
    )

    if candidate_token.pos_ in exclusions:

        debug_log(
            "STEP 2 RESULT",
            f"{word_lower} => REAL_WORD => FALSE"
        )

        return False

    debug_log(
        "STEP 2 RESULT",
        f"{word_lower} => FILLER => TRUE"
    )

    return True


# ─────────────────────────────────────────────────────────────
# STEP 3 — REPETITION DETECTION
# ─────────────────────────────────────────────────────────────

def detect_repetitions(text):

    debug_log("STEP 3 - REPETITION INPUT", text)

    repetitions = []

    words = re.findall(r"\b\w+\b", text.lower())

    debug_log("STEP 3 - TOKENIZED WORDS", words)

    for i in range(1, len(words)):

        current_word = words[i]
        previous_word = words[i - 1]

        if current_word == previous_word:

            repetitions.append(current_word)

            debug_log(
                "STEP 3 - REPETITION FOUND",
                {
                    "word": current_word,
                    "index": i
                }
            )

    debug_log("STEP 3 RESULT - REPETITIONS", repetitions)

    return repetitions


# ─────────────────────────────────────────────────────────────
# STEP 4 — MAIN DETECTION
# ─────────────────────────────────────────────────────────────

def detect_fillers(text: str):

    debug_log("STEP 4 - START DETECTION", text)

    chunks = _presplit(text)

    confirmed_fillers = []
    counts = {}

    prose_so_far = ""

    for chunk in chunks:

        debug_log("STEP 4 - PROCESSING CHUNK", chunk)

        if not chunk["is_candidate"]:

            prose_so_far += chunk["text"]

            debug_log(
                "STEP 4 - UPDATED PROSE",
                prose_so_far
            )

            continue

        word_lower = chunk["text"].lower().strip()

        is_valid = _validate_candidate(
            word_lower,
            prose_so_far
        )

        debug_log(
            "STEP 4 - VALIDATION RESULT",
            {
                "word": word_lower,
                "is_filler": is_valid
            }
        )

        if is_valid:

            counts[word_lower] = counts.get(word_lower, 0) + 1

            confirmed_fillers.append({
                "word": word_lower,
                "start_char": chunk["start"],
                "end_char": chunk["end"],
            })

            debug_log(
                "STEP 4 - CONFIRMED FILLER",
                confirmed_fillers[-1]
            )

        else:

            prose_so_far += chunk["text"]

            debug_log(
                "STEP 4 - TREATED AS NORMAL WORD",
                prose_so_far
            )

    # REPETITION DETECTION
    repetitions = detect_repetitions(text)

    # BUILD SEGMENTS
    debug_log("STEP 4 - BUILDING SEGMENTS", confirmed_fillers)

    segments = []

    cursor = 0

    for f in sorted(confirmed_fillers, key=lambda x: x["start_char"]):

        if f["start_char"] < cursor:
            continue

        if f["start_char"] > cursor:

            normal_segment = text[cursor:f["start_char"]]

            segments.append({
                "text": normal_segment,
                "isFiller": False
            })

            debug_log(
                "STEP 4 - NORMAL SEGMENT",
                normal_segment
            )

        filler_segment = text[f["start_char"]:f["end_char"]]

        segments.append({
            "text": filler_segment,
            "isFiller": True
        })

        debug_log(
            "STEP 4 - FILLER SEGMENT",
            filler_segment
        )

        cursor = f["end_char"]

    if cursor < len(text):

        remaining = text[cursor:]

        segments.append({
            "text": remaining,
            "isFiller": False
        })

        debug_log(
            "STEP 4 - REMAINING SEGMENT",
            remaining
        )

    final_result = {
        "fillerCounts": counts,
        "segments": segments,
        "repetitions": repetitions,
    }

    debug_log("FINAL OUTPUT", final_result)

    return final_result

_GEMINI_SYSTEM = (
    "You are a speech transcription corrector. "
    "A speech-to-text engine has over-cleaned a transcript by removing filler "
    "words and hesitation markers that the speaker actually said. "
    "Your job: restore the natural spoken fillers so the transcript sounds "
    "exactly like real spontaneous speech. "
    "Rules: "
    "Insert fillers (um, uh, hmm, like, you know, I mean, so, basically, "
    "right, actually, kind of, sort of, okay, well, er, ah) where a real "
    "speaker would naturally pause or hesitate. "
    "Restore repeated words if the original likely had word repetitions "
    "(e.g. 'I I think' or 'the the problem'). "
    "Keep every real word intact — never remove or reorder content words. "
    "Output ONLY the restored transcript. No explanations, no markdown."
)

# Few-shot turns formatted for Gemini's `contents` array
_GEMINI_FEW_SHOTS = [
    {
        "role": "user",
        "parts": [{"text": (
            "Cleaned: So we decided to go with the new approach because it "
            "seemed more efficient and the team agreed."
        )}],
    },
    {
        "role": "model",
        "parts": [{"text": (
            "So um we decided to go with the new approach because it like "
            "seemed more efficient and uh the team agreed."
        )}],
    },
    {
        "role": "user",
        "parts": [{"text": (
            "Cleaned: I think the issue is with the database connection and "
            "we need to fix it before the demo."
        )}],
    },
    {
        "role": "model",
        "parts": [{"text": (
            "I I think the issue is like with the database connection and "
            "you know we need to fix it before the demo."
        )}],
    },
]

_GEMINI_MODEL = "gemini-1.5-flash-8b"
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{_GEMINI_MODEL}:generateContent"
)


async def restore_fillers_via_llm(text: str) -> str:
    """
    Optional second-pass: uses Gemini 1.5 Flash 8B (free) to re-insert
    natural fillers that Sarvam stripped.
    Returns original text if GEMINI_API_KEY is unset or if the call fails.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not gemini_key:
        debug_log("LLM RESTORE", "GEMINI_API_KEY not set — skipping LLM pass")
        return text

    contents = _GEMINI_FEW_SHOTS + [
        {
            "role": "user",
            "parts": [{"text": f"Cleaned: {text}"}],
        }
    ]

    payload = {
        "system_instruction": {"parts": [{"text": _GEMINI_SYSTEM}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _GEMINI_URL,
                params={"key": gemini_key},
                headers={"Content-Type": "application/json"},
                json=payload,
            )

        debug_log("GEMINI STATUS", resp.status_code)

        if resp.status_code != 200:
            debug_log("GEMINI ERROR", resp.text)
            return text

        restored = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        debug_log("LLM RESTORED TRANSCRIPT", restored)
        return restored if restored else text

    except Exception as exc:
        debug_log("LLM RESTORE EXCEPTION", str(exc))
        return text


# ─────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
# TRANSCRIBE
# ─────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    api_key: str = Form(...),
):

    debug_log("API REQUEST STARTED", file.filename)

    audio_bytes = await file.read()

    debug_log(
        "AUDIO INFO",
        {
            "filename": file.filename,
            "size_bytes": len(audio_bytes)
        }
    )

    # ── Few-shot style hint for Sarvam (prompt acts as a prefix the model
    #    tries to *continue*, so we show it examples of verbatim speech) ──
    FEW_SHOT_PROMPT = (
        "um so I was like thinking about it and uh basically I mean it makes "
        "sense right like you know what I mean hmm yeah so uh we should uh "
        "actually try it I I mean it it could work well you know okay so "
        "basically yeah let's do it um"
    )

    async with httpx.AsyncClient(timeout=120) as client:

        debug_log("CALLING SARVAM API", "STARTED")

        r = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": api_key},
            files={
                "file": (
                    "recording.webm",
                    io.BytesIO(audio_bytes),
                    "audio/webm"
                )
            },
            data={
                "model": "saaras:v3",
                "mode": "verbatim",
                "with_timestamps": True,
                "diarize": False,
                "prompt": FEW_SHOT_PROMPT,
            },
        )

    debug_log("SARVAM STATUS CODE", r.status_code)

    if r.status_code != 200:

        debug_log("SARVAM ERROR", r.text)

        return {
            "error": r.text,
            "transcript": "",
            "fillerCounts": {},
            "segments": [],
            "repetitions": [],
        }

    data = r.json()

    debug_log("RAW SARVAM RESPONSE", data)

    raw = (
        data.get("transcript")
        or data.get("text")
        or ""
    ).strip()

    debug_log("RAW TRANSCRIPT", raw)

    # REMOVE INDIC SCRIPTS
    text = re.sub(
        r"[\u0B80-\u0BFF]+",
        "",
        raw
    ).strip()

    debug_log("CLEANED TRANSCRIPT", text)

    text = await restore_fillers_via_llm(text)
    debug_log("POST-LLM TRANSCRIPT", text)
    # ───────────────────────────────────────────────────────────────────

    result = detect_fillers(text)

    final_response = {
        "transcript": text,
        **result
    }

    debug_log("FINAL API RESPONSE", final_response)

    return final_response