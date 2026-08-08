"""
Vastu Shastra AI Analyzer
-------------------------
A Streamlit prototype that takes floor plan descriptions (free text or a
structured room/direction table) and returns a Vastu analysis grounded in
a fixed knowledge base, generated via a Hugging Face-hosted LLM.

Run:
    streamlit run app.py

Required secret (either works):
    - Environment variable HF_TOKEN
    - Streamlit secret: st.secrets["HF_TOKEN"]  (recommended for HF Spaces)
"""

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vastu_ai")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
AVAILABLE_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct:featherless-ai",
    "meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
    "Qwen/Qwen2.5-7B-Instruct:featherless-ai",
]

# Vision-capable (image-text-to-text) models. IMPORTANT: Featherless AI (the
# provider pinned below, matching your working text-model setup) only passes
# image input through for Gemma- and Mistral-class models - NOT Qwen-VL,
# despite Qwen-VL being labeled "image-text-to-text" on its model page. That
# mismatch is what caused the earlier 400 errors. Gemma 3 is confirmed to
# have Featherless AI as an Inference Provider.
#
# ONE-TIME SETUP: google/gemma-3-27b-it is a gated model. Before it will
# work, log into huggingface.co with the account tied to your HF_TOKEN,
# open https://huggingface.co/google/gemma-3-27b-it, and accept Google's
# usage license - otherwise every request 403s regardless of the code.
VISION_MODELS = [
    "google/gemma-3-27b-it:featherless-ai",
]

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 45
MAX_INPUT_CHARS = 6000

MAX_IMAGE_MB = 8
MAX_IMAGE_DIMENSION = 1024  # longest side, px - kept modest so the base64
                            # payload stays small; large payloads are more
                            # likely to get truncated on restrictive/proxied
                            # networks, which surfaces as a confusing
                            # JSONDecodeError deep inside the HTTP client.
JPEG_QUALITY = 78
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

DIRECTIONS_8 = [
    "North", "North-East", "East", "South-East",
    "South", "South-West", "West", "North-West",
]

# Which compass direction the TOP of the uploaded image points to. Floor
# plan images almost never encode compass orientation on their own (unless
# there's a visible north arrow), so this must come from the user.
TOP_OF_IMAGE_OPTIONS = DIRECTIONS_8

# --------------------------------------------------------------------------
# Knowledge base (condensed from the Vastu reference doc). This is injected
# into the system prompt so the model's output is grounded in *your* rules
# rather than whatever generic Vastu content it picked up in training.
# --------------------------------------------------------------------------
VASTU_KNOWLEDGE_BASE = """
CORE PHILOSOPHY
Vastu Shastra maps human activities to spatial/magnetic/solar alignment.
Two foundations: the Vastu Purusha Mandala (energy grid of a space) and
Panchatatva, the five elements (Water, Fire, Earth, Air, Space), each
governing a direction. A compliant plan matches room function to the
element of its zone.

STANDARD 8-DIRECTION SYSTEM (45-degree slices)
- North-East (Ishan) / Water: IDEAL - pooja room, underground water tanks,
  open space. DEFECTS - toilets, kitchens, septic tanks, heavy stairs.
- South-East (Agni) / Fire: IDEAL - kitchen, electrical panels, heaters.
  DEFECTS - underground water tanks, main entrance.
- South-West (Nairutya) / Earth: IDEAL - master bedroom, heavy storage,
  cash lockers. DEFECTS - main entrance, underground water, toilets.
- North-West (Vayavya) / Air: IDEAL - guest rooms, finished goods,
  septic tanks. DEFECTS - master bedroom.
- Center (Brahmasthan) / Space: IDEAL - open courtyard, living area.
  DEFECTS - pillars, load-bearing walls, toilets.

16-ZONE MAHAVASTU FRAMEWORK (22.5-degree slices) - use for finer analysis
- North-North-East (NNE): Health & Immunity. Ideal for medicines; defective
  if a kitchen/fire element is present.
- East-South-East (ESE): Analysis. Ideal for study; a bedroom here causes
  anxiety.
- South-South-West (SSW): Disposal. Best location for toilets/dustbins;
  disastrous for financial lockers.
- West-North-West (WNW): Detoxification. Ideal for washing machines.
(Other 16-zone points follow the same logic as their nearest 8-direction
parent unless the user's data lets you be more precise.)

REMEDIAL PRINCIPLE
Prefer remedies WITHOUT demolition:
- Color Therapy: paint/mat in the element's color to rebalance a zone
  (e.g. red tones in South-East).
- Metal Strips: copper/brass/steel strips to block negative flow, e.g.
  around a misplaced toilet.
- Objects & Mirrors: mirrors to "extend" a cut corner; heavy objects to
  ground a zone.
"""

SYSTEM_PROMPT = f"""You are a Vastu Shastra consultant AI operating strictly on the knowledge \
base below. Do not invent rules that contradict it, and do not rely on Vastu claims outside it \
unless the user's data requires a direction not covered, in which case say so explicitly.

KNOWLEDGE BASE:
{VASTU_KNOWLEDGE_BASE}

TASK:
You may receive the floor plan as (a) typed text mapping rooms to compass directions, or (b) an \
image of a floor plan plus a stated compass direction for the TOP of the image. You must:
1. Identify each room/element and the compass direction it falls in.
   - For an IMAGE: visually identify rooms/fixtures (look for labels, door/window breaks, stove
     icons, toilet fixtures, staircases, the main entrance). Use the stated "top of image" direction
     to convert each element's position (top/bottom/left/right/corners) into a compass direction.
     If a label or feature is illegible or ambiguous, list it as "unclear" rather than guessing.
2. Flag any DEFECTS by matching against the knowledge base (ideal vs defective placements).
3. For each defect, explain the impact briefly and give a specific non-demolition remedy from the \
knowledge base (color therapy, metal strips, mirrors/objects) - do not suggest structural changes.
4. If no defects are found for a section, say so; do not fabricate problems.
5. Note explicitly if information is missing, illegible, or ambiguous (e.g. direction not stated, \
a room label you could not read) rather than guessing.

OUTPUT FORMAT:
Respond with ONLY a JSON object (no markdown fences, no prose outside the JSON) matching this shape:
{{
  "summary": "one or two sentence overall verdict",
  "detected_layout": [
    {{"element": "...", "direction": "...", "confidence": "high|medium|low"}}
  ],
  "defects": [
    {{"location": "...", "issue": "...", "impact": "...", "remedy": "..."}}
  ],
  "compliant_zones": ["short strings noting what is already correct"],
  "notes": "any caveats, missing info, illegible elements, or assumptions you made"
}}
"detected_layout" lets the user verify you read the plan correctly before trusting the defects -
always populate it when analyzing an image. If there are no defects, return an empty list for
"defects" and explain why in "summary".
"""


# --------------------------------------------------------------------------
# Data model for a parsed response
# --------------------------------------------------------------------------
@dataclass
class VastuReport:
    summary: str = ""
    detected_layout: list = field(default_factory=list)
    defects: list = field(default_factory=list)
    compliant_zones: list = field(default_factory=list)
    notes: str = ""
    raw_text: str = ""
    parse_ok: bool = True


# --------------------------------------------------------------------------
# Client / API helpers
# --------------------------------------------------------------------------
def get_hf_token() -> Optional[str]:
    """Look in st.secrets first (recommended for HF Spaces / Streamlit Cloud),
    then fall back to an environment variable."""
    token = None
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        pass
    if not token:
        token = os.environ.get("HF_TOKEN")
    return token


@st.cache_resource(show_spinner=False)
def get_client(token: str) -> OpenAI:
    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=token, timeout=REQUEST_TIMEOUT_SECONDS)


def load_and_validate_image(uploaded_file) -> tuple[Optional[Image.Image], Optional[str]]:
    """Open + validate an uploaded image. Returns (PIL Image, error_message)."""
    if uploaded_file is None:
        return None, "No file uploaded."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        return None, f"Image is {size_mb:.1f} MB - please upload something under {MAX_IMAGE_MB} MB."

    try:
        img = Image.open(uploaded_file)
        img.load()  # force decode now so corrupt files fail here, not later
    except UnidentifiedImageError:
        return None, "Could not read this file as an image. Please upload a PNG, JPEG, or WEBP."
    except Exception as e:
        return None, f"Failed to open image: {e}"

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    return img, None


def encode_image_to_data_url(img: Image.Image) -> str:
    """Downscale (if needed) and encode a PIL image as a base64 JPEG data URL."""
    resized = img.copy()
    resized.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    if resized.mode != "RGB":
        resized = resized.convert("RGB")

    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_user_content(user_text: str, image_data_url: Optional[str]):
    """Build the `content` field for the user message - plain string for text-only,
    a multimodal content list when an image is attached."""
    if not image_data_url:
        return f"Analyze this floor plan data:\n{user_text}"

    return [
        {"type": "text", "text": f"Analyze this floor plan image.\n{user_text}"},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]


def _is_capacity_error(e: APIStatusError) -> bool:
    """5xx / 'temporarily at capacity' errors are worth retrying or falling
    back to another model - unlike a 400 (bad request) or 401 (auth)."""
    status = getattr(e, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    body = str(getattr(e, "body", "") or "") + str(e)
    return "capacity" in body.lower() or "overloaded" in body.lower()


def call_llm_with_retries(
    client: OpenAI,
    model: str,
    user_content: str,
    temperature: float,
    image_data_url: Optional[str] = None,
) -> str:
    """Call the chat completion endpoint with retries/backoff for transient errors,
    including 5xx/capacity errors from the provider (not just network issues)."""
    last_error = None
    message_content = build_user_content(user_content, image_data_url)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message_content},
                ],
            )
            content = completion.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Model returned an empty response.")
            return content
        except (APITimeoutError, APIConnectionError, RateLimitError, json.JSONDecodeError) as e:
            last_error = e
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning("Transient error on attempt %d/%d: %s. Retrying in %.1fs.",
                            attempt, MAX_RETRIES, e, wait)
            if attempt < MAX_RETRIES:
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Got a malformed/truncated response after {MAX_RETRIES} attempts "
                f"(last error: {e}). This can happen on networks that interfere with large "
                "requests (corporate/college proxies, VPNs)."
            ) from e
        except APIStatusError as e:
            if _is_capacity_error(e) and attempt < MAX_RETRIES:
                last_error = e
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning("Capacity error on attempt %d/%d for model %s: %s. Retrying in %.1fs.",
                                attempt, MAX_RETRIES, model, e, wait)
                time.sleep(wait)
                continue
            # Non-transient (4xx auth/model errors etc.), or out of retries - don't loop further
            logger.error("API status error: %s", e)
            raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")


def call_llm_with_fallback(
    client: OpenAI,
    models: list,
    user_content: str,
    temperature: float,
    image_data_url: Optional[str] = None,
) -> tuple[str, str]:
    """Try a list of models in order, moving to the next one on ANY error
    except auth (401) - a 400/404/5xx from one model/provider doesn't mean
    another model in the list will fail the same way. Returns
    (response_text, model_that_succeeded). Raises the last error if every
    model in the list fails."""
    last_error = None
    for model in models:
        try:
            result = call_llm_with_retries(client, model, user_content, temperature, image_data_url)
            return result, model
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 401:
                raise  # bad/expired token - every model will fail the same way
            logger.warning("Model %s failed (status %s): %s. Trying next fallback if any.",
                            model, getattr(e, "status_code", "?"), e)
            last_error = e
            continue
        except RuntimeError as e:
            logger.warning("Model %s exhausted retries: %s. Trying next fallback if any.", model, e)
            last_error = e
            continue
    raise last_error if last_error else RuntimeError("No models were attempted.")


def parse_report(raw_text: str) -> VastuReport:
    """Try to parse the model's JSON output; fall back to raw text gracefully."""
    text = raw_text.strip()
    # Strip accidental markdown fences if the model adds them anyway.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        return VastuReport(
            summary=data.get("summary", ""),
            detected_layout=data.get("detected_layout", []) or [],
            defects=data.get("defects", []) or [],
            compliant_zones=data.get("compliant_zones", []) or [],
            notes=data.get("notes", ""),
            raw_text=raw_text,
            parse_ok=True,
        )
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("Could not parse model output as JSON: %s", e)
        return VastuReport(raw_text=raw_text, parse_ok=False)


def validate_input(text: str) -> Optional[str]:
    """Return an error message if invalid, else None."""
    if not text or not text.strip():
        return "Please enter some floor plan data to analyze."
    if len(text) > MAX_INPUT_CHARS:
        return f"Input is too long ({len(text)} chars). Please keep it under {MAX_INPUT_CHARS} characters."
    return None


def structured_rows_to_text(df: pd.DataFrame) -> str:
    """Convert the structured room/direction table into text the LLM can read."""
    rows = []
    for _, row in df.iterrows():
        room = str(row.get("Room / Element", "")).strip()
        direction = str(row.get("Direction", "")).strip()
        if room and direction and direction != "-- select --":
            rows.append(f"- {room} is in the {direction}")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Vastu AI Analyzer", page_icon="🏠", layout="centered")
st.title("🏠 Vastu Shastra AI Analyzer")
st.caption("Prototype - analysis is grounded in a fixed Vastu knowledge base, not general model knowledge.")

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                             help="Lower = more consistent/rule-following. Recommended: stay low for this use case.")
    st.divider()
    st.markdown(
        "**Setup**: set `HF_TOKEN` as a Streamlit secret (`.streamlit/secrets.toml` or "
        "Space Secrets) or as an environment variable."
    )
    st.caption(
        "Model choice is on each tab: text tabs use a text model, the image tab needs a "
        "vision (image-text-to-text) model."
    )

tab_free, tab_structured, tab_image = st.tabs(["Free text", "Structured table", "Upload image"])

with tab_free:
    default_placeholder = (
        "e.g.,\n"
        "- Kitchen is in the North-East\n"
        "- Master Bedroom is in the South-West\n"
        "- Main Entrance faces South\n"
    )
    free_text_data = st.text_area(
        "Floor Plan Data", height=200, placeholder=default_placeholder, key="free_text"
    )
    model_free = st.selectbox("Model", AVAILABLE_MODELS, index=0, key="model_free")
    analyze_free = st.button("Analyze Vastu", type="primary", key="analyze_free_btn")

with tab_structured:
    st.caption("Add one row per room/element and pick its compass direction.")
    default_df = pd.DataFrame(
        [{"Room / Element": "", "Direction": "-- select --"}]
    )
    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        column_config={
            "Room / Element": st.column_config.TextColumn(required=True),
            "Direction": st.column_config.SelectboxColumn(
                options=["-- select --"] + DIRECTIONS_8, required=True
            ),
        },
        use_container_width=True,
        key="structured_editor",
    )
    model_structured = st.selectbox("Model", AVAILABLE_MODELS, index=0, key="model_structured")
    analyze_structured = st.button("Analyze Vastu", type="primary", key="analyze_structured_btn")

with tab_image:
    st.caption(
        "Upload a floor plan image (PNG, JPEG, or WEBP). Floor plans rarely encode compass "
        "direction on their own, so tell us which way the top of the image faces - use the "
        "north arrow on the plan if it has one."
    )
    uploaded_image = st.file_uploader(
        "Floor plan image", type=ALLOWED_IMAGE_TYPES, key="floor_plan_image"
    )
    top_direction = st.selectbox(
        "The TOP of this image points toward...", TOP_OF_IMAGE_OPTIONS, index=0, key="top_direction"
    )
    extra_context = st.text_area(
        "Optional: anything not visible/legible in the image (e.g. 'main entrance is on the "
        "south wall', 'the small room top-left is a store room')",
        height=100, key="image_extra_context",
    )
    model_image = st.selectbox("Vision model", VISION_MODELS, index=0, key="model_image")

    preview_img = None
    if uploaded_image is not None:
        preview_img, preview_err = load_and_validate_image(uploaded_image)
        if preview_err:
            st.warning(preview_err)
        else:
            st.image(preview_img, caption="Preview", use_container_width=True)

    analyze_image = st.button("Analyze Vastu", type="primary", key="analyze_image_btn")

trigger = False
payload_text = ""
model = AVAILABLE_MODELS[0]
image_data_url = None

if analyze_free:
    trigger = True
    payload_text = free_text_data
    model = model_free

if analyze_structured:
    trigger = True
    payload_text = structured_rows_to_text(edited_df)
    model = model_structured

if analyze_image:
    if uploaded_image is None:
        st.warning("Please upload a floor plan image first.")
    else:
        img, img_err = load_and_validate_image(uploaded_image)
        if img_err:
            st.warning(img_err)
        else:
            trigger = True
            model = model_image
            image_data_url = encode_image_to_data_url(img)
            payload_text = (
                f"The top of the image points toward: {top_direction}.\n"
                f"Additional context from the user: {extra_context.strip() or 'none provided.'}"
            )

if trigger:
    error_msg = validate_input(payload_text)
    if error_msg:
        st.warning(error_msg)
    else:
        hf_token = get_hf_token()
        if not hf_token:
            st.error(
                "Authentication error: HF_TOKEN is not set. Add it as a Streamlit secret "
                "or as an environment variable and reload the app."
            )
        else:
            try:
                client = get_client(hf_token)
                spinner_msg = (
                    "Reading the floor plan image..." if image_data_url
                    else "Consulting the Vastu rules engine..."
                )
                with st.spinner(spinner_msg):
                    if image_data_url:
                        # Try the chosen vision model first, fall back to the others
                        # in VISION_MODELS if it's at capacity.
                        fallback_order = [model] + [m for m in VISION_MODELS if m != model]
                        raw_response, model_used = call_llm_with_fallback(
                            client, fallback_order, payload_text, temperature,
                            image_data_url=image_data_url,
                        )
                        if model_used != model:
                            st.info(f"'{model}' was at capacity - used '{model_used}' instead.")
                    else:
                        raw_response = call_llm_with_retries(client, model, payload_text, temperature)
                report = parse_report(raw_response)

                st.success("Analysis complete")

                if report.parse_ok:
                    st.markdown("### Summary")
                    st.write(report.summary or "_No summary returned._")

                    if report.detected_layout:
                        st.markdown("### Detected Layout (verify this before trusting the defects)")
                        st.dataframe(
                            pd.DataFrame(report.detected_layout),
                            use_container_width=True, hide_index=True,
                        )

                    st.markdown("### Defects & Remedies")
                    if report.defects:
                        for d in report.defects:
                            with st.container(border=True):
                                st.markdown(f"**{d.get('location', 'Unknown location')}**")
                                st.write(f"Issue: {d.get('issue', '-')}")
                                st.write(f"Impact: {d.get('impact', '-')}")
                                st.write(f"Remedy: {d.get('remedy', '-')}")
                    else:
                        st.info("No defects identified.")

                    if report.compliant_zones:
                        st.markdown("### Already Compliant")
                        for z in report.compliant_zones:
                            st.write(f"- {z}")

                    if report.notes:
                        st.markdown("### Notes")
                        st.caption(report.notes)

                    with st.expander("Raw model output (JSON)"):
                        st.code(report.raw_text, language="json")
                else:
                    st.warning(
                        "The model's response wasn't valid JSON, showing raw output instead."
                    )
                    st.markdown(report.raw_text)

            except APIStatusError as e:
                status = getattr(e, "status_code", "unknown")
                if status == 401:
                    st.error("Authentication failed. Check that HF_TOKEN is valid and not expired.")
                elif status == 403:
                    st.error(
                        f"Access denied for '{model}' (status 403). This model is gated - log in "
                        "to huggingface.co with the account tied to your HF_TOKEN, open the "
                        "model's page, and accept the usage license, then try again."
                    )
                elif status == 404:
                    st.error(f"Model '{model}' was not found on the router. Try a different model.")
                elif status == 429:
                    st.error("Rate limited by the API. Wait a moment and try again.")
                elif status and isinstance(status, int) and status >= 500:
                    st.error(
                        "All models tried are currently at capacity on Hugging Face's providers. "
                        "This is on their end - wait a minute and try again."
                    )
                elif status == 400 and "not supported by any provider" in str(e).lower():
                    st.error(
                        f"'{model}' isn't enabled on your Hugging Face account's Inference "
                        "Providers. Enable a provider for this model at "
                        "https://huggingface.co/settings/inference-providers, or switch to a "
                        "model/provider combo you've already confirmed works."
                    )
                elif status == 400 and image_data_url:
                    st.error(
                        "Every vision model tried rejected this request. This could be an "
                        "unsupported image format/size, or a provider-specific issue rather than "
                        "the model lacking vision support."
                    )
                else:
                    st.error(f"API returned an error (status {status}): {e}")
                with st.expander("Error detail (for debugging)"):
                    st.code(str(e))
                logger.exception("APIStatusError during analysis")
            except (APITimeoutError, APIConnectionError, RateLimitError, RuntimeError) as e:
                st.error(f"Could not reach the model after retries: {e}")
                logger.exception("Retryable error exhausted during analysis")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logger.exception("Unexpected error during analysis")