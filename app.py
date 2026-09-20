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
Here is an expanded, descriptive, and scholarly edition of your knowledge base, grounded directly in the four classic treatises (*Mayamata* Vols. I & II, Vibhuti Chakrabarti’s *Indian Architectural Theory*, and Sashikala Ananth’s *The Penguin Guide to Vaastu*).

---

### 1. CORE PHILOSOPHY & THEORETICAL FOUNDATIONS

* **Ontological Distinction Between *Vastu* and *Vaastu***:
  * ***Vastu*** refers to the unmanifest and manifest subtle life energy (*Jivatman* / *Praana*) residing in all matter, earth, and physical structures. 
  * ***Vaastu*** denotes the physical site, spatial enclosure, or support (*Bhoomi* / *Prasada*) upon which this life energy is anchored and expressed.

* **The Three Classical Principles of Design (*Trisutra*)**:
  * ***Bhogadyam***: Functional utility, structural comfort, and spatial efficiency.
  * ***Sukha Darsham***: Aesthetic visual balance, geometric proportion, and spatial beauty.
  * ***Ramya***: The inner contentment, emotional resonance, and feeling of well-being that evokes a sympathetic vibration between the built form and the user.

* **The *Vastu Purusha Mandala* (The Master Cosmic Grid)**:
  * The *Vastu Purusha Mandala* is the core design grid representing the Cosmic Man (*Purusha*) pressed face-downward onto the consecrated ground by various divinities. 
  * While 32 distinct grid layouts exist, residential planning primarily utilizes the **64-square (*Manduka*)** and **81-square (*Paramashayika*)** square grids.
  * **Anthropomorphic Alignment**: The *Purusha* lies diagonally across the grid: the **head** rests in the North-East (*Ishana*), the **feet** meet in the South-West (*Nairutya/Pitri*), the **joints/limbs** lie in the South-East and North-West, and the **navel/heart** occupies the central Brahmasthana.

* **Panchamahabhutas (The Five Primary Elements)**:
  * Architecture is conceived as a microcosm balancing the five universal elements: **Water** (*Apah*), **Fire** (*Agni*), **Earth** (*Prithvi*), **Air** (*Vayu*), and **Space/Ether** (*Akasha*).

* **System of Relative Proportion (*Pramana*) & *Ayadi* Calculations**:
  * **Relative Proportioning**: Dimensions originate from human scale units—*Angula* (finger width ~1.9 cm), *Hasta* (cubit = 24 Angulas), and *Danda* (rod = 4 Hastas). The width (*W*) of a plinth is the fundamental module from which building height, length, wall thickness, and opening sizes are derived.
  * **Ayadi Formulae (*Shadayadi*)**: Astrological mathematical checks calculated from the plinth area or perimeter to ensure spatial-temporal resonance between the householder's birth star (*Janma Nakshatra*) and the building. The six core checks include:
    1. ***Aya*** (Income/Directional Propensity): Remainder of \\(\text{Area} / 8\\). Yields 8 directional types (*Dhwaja*, *Singha*, *Vrishabha*, *Gaja*, etc.); odd remainders are auspicious.
    2. ***Vyaya*** (Expenditure/Debt): Remainder of \\((\text{Area} \times 3) / 8\\). *Aya* must always exceed *Vyaya* to ensure prosperity.
    3. ***Nakshatra*** (Lunar Mansion of the House): Remainder of \\((\text{Area} \times 8) / 27\\).
    4. ***Tithi*** (Lunar Date): Remainder of \\((\text{Area} \times 8) / 15\\).
    5. ***Ayu*** (Vital Lifespan of Building): Remainder of \\((\text{Area} \times 8) / 120\\).
    6. ***Yoni*** (Directional Energy Flow): Evaluates spatial orientation; odd remainders (1=East/Dhwaja, 3=South/Singha, 5=West/Vrishabha, 7=North/Gaja) are favorable.

---

### 2. ENHANCED 8-DIRECTIONAL COSMOLOGY & DIURNAL SUN CYCLE

The 24-hour diurnal cycle of the sun establishes the spatial zoning of domestic functions across the eight cardinal directions:

```
                     NORTH-WEST (Vayavya)           NORTH (Soma/Kubera)           NORTH-EAST (Ishana)
                     Air / Moon                    Water / Mercury               Water / Jupiter
                     9 PM - 12 Midnight            12 Midnight - 3 AM            3 AM - 6 AM
                     [Guest, Granary, Toilets]     [Treasury, Medicine]          [Meditation, Puja, Water]
                                    \                        |                        /
                                     \                       |                       /
          WEST (Varuna)               \                      |                      /            EAST (Surya/Aditya)
          Air / Saturn                 \                     |                     /             Fire/Light / Sun
          6 PM - 9 PM                   +------------- BRAHMASTHANA -------------+              6 AM - 9 AM
          [Dining, Study]               |               Space / Ether            |              [Main Entrance, Bath]
                                       /               Open Courtyard            \
                                      /                      |                    \
                                     /                       |                     \
                     SOUTH-WEST (Nairutya/Pitri)    SOUTH (Yama)                  SOUTH-EAST (Agneya/Agni)
                     Earth / Rahu                   Earth / Mars                  Fire / Venus
                     3 PM - 6 PM                    12 Noon - 3 PM                9 AM - 12 Noon
                     [Master Bed, Heavy Storage]    [Bedrooms, Armoury]           [Kitchen, Heaters, Ghee]
```

* **North-East (*Ishana*) — Water Zone**:
  * **Attributes**: Governed by Shiva (*Ishana*) and Planet Jupiter (*Guru*); represents absolute purity, spirituality, and the head of the *Vastu Purusha*.
  * **Diurnal Time Slice**: 3:00 AM – 6:00 AM (*Brahma Muhurta*).
  * **Ideal Functions**: Prayer/puja room, meditation, underground water storage/wells, open unencumbered space.
  * **Defects**: Heavy construction, toilets, kitchens, septic tanks, or cut corners (causes spiritual decay, family discord, and loss of male progeny).

* **East (*Surya / Aditya*) — Light/Fire Zone**:
  * **Attributes**: Governed by Indra (power) and the Sun (*Surya*); rules vitality, royal favor, blood circulation, and soul.
  * **Diurnal Time Slice**: 6:00 AM – 9:00 AM (Sunrise).
  * **Ideal Functions**: Main entrance/access, bathrooms (for morning sun exposure), open verandas, multipurpose rooms.
  * **Defects**: High solid walls or absence of openings blocking beneficial morning solar rays.

* **South-East (*Agneya / Agni*) — Fire Zone**:
  * **Attributes**: Governed by Agni (sacred fire) and Planet Venus (*Shukra*); rules domestic energy, women's health, conjugal felicity, and cooking.
  * **Diurnal Time Slice**: 9:00 AM – 12:00 PM.
  * **Ideal Functions**: Kitchen, electrical switchboards, generators, boilers, storage of ghee/oil.
  * **Defects**: Underground water tanks, main entrance, or bedrooms (causes fear of fire, general relaxation/loss of stimulus, and female health issues).

* **South (*Yama / Dakshina*) — Earth Zone**:
  * **Attributes**: Governed by Yama (justice/death) and Planet Mars (*Mangal*); rules strength, stamina, legal justice, and physical stability.
  * **Diurnal Time Slice**: 12:00 PM – 3:00 PM.
  * **Ideal Functions**: Bedrooms, heavy storage, high structural mass.
  * **Defects**: Underground water sources, main entrance, or downward site slope (causes mortality and legal disputes).

* **South-West (*Nairutya / Pitri*) — Earth/Ancestral Zone**:
  * **Attributes**: Governed by Nirriti/Pitri (ancestors) and Planet Rahu; represents stability, family lineage, and the feet of the *Vastu Purusha*.
  * **Diurnal Time Slice**: 3:00 PM – 6:00 PM.
  * **Ideal Functions**: Master bedroom, heavy storage, cash lockers, overhead water tanks, highest roof level.
  * **Defects**: Underground water tanks, main entrance, open courtyards, toilets, or cut corners (causes severe financial drain, loss of longevity, and servitude).

* **West (*Varuna / Paschima*) — Air/Water Zone**:
  * **Attributes**: Governed by Varuna (cosmic waters) and Planet Saturn (*Shani*); rules longevity, retentive strength, discipline, and constant wealth.
  * **Diurnal Time Slice**: 6:00 PM – 9:00 PM (Sunset).
  * **Ideal Functions**: Dining hall, study rooms, study libraries.
  * **Defects**: Absence of western structural wings or lower elevation than East (leads to loss of wealth retention and chronic depression).

* **North-West (*Vayavya / Vayu*) — Air Zone**:
  * **Attributes**: Governed by Vayu (wind) and the Moon (*Chandra*); rules mobility, social relationships, mental peace, and distribution.
  * **Diurnal Time Slice**: 9:00 PM – 12:00 Midnight.
  * **Ideal Functions**: Guest bedrooms, granaries, finished goods storage, septic tanks, marriageable daughters' rooms.
  * **Defects**: Master bedroom (causes mental instability, futile wandering, and inability to settle).

* **North (*Soma / Kubera*) — Water/Wealth Zone**:
  * **Attributes**: Governed by Kubera (wealth) and Planet Mercury (*Budha*); rules financial influx, trade, communication, and health.
  * **Diurnal Time Slice**: 12:00 Midnight – 3:00 AM.
  * **Ideal Functions**: Treasuries, cash safes, medicine storage, open verandas, northern water bodies.
  * **Defects**: Heavy solid walls, toilets, or garbage dumps (blocks wealth influx and causes financial stagnation).

* **Center (*Brahmasthana*) — Ether/Space Zone**:
  * **Attributes**: Governed by Brahma (creator); represents the unencumbered cosmic navel/heart (*Mahamarma*).
  * **Ideal Functions**: Open central courtyard (*Angana*), light-filled atrium, sacred *Tulasi* plant shrine.
  * **Defects**: Load-bearing pillars, heavy walls, staircases, toilets, or water tanks directly over the intersection of main diagonal lines (*Vamsha/Sira*) (causes destruction of the household and instant physical/financial collapse).

---

### 3. MICRO-ZONING, CONCENTRIC GRIDS (*PADA VINYASA*) & DEITY MAPS

To refine functional placement, the *Mandala* is structured concentrically from the center outward into four distinct energy belts (*Veethis/Padams*):

1. ***Brahmapadam*** (Central Core — 9 squares in 81-grid): Sacred Ether zone; must remain open to the sky as a central courtyard (*Angana*).
2. ***Daivikapadam*** (Divine Inner Ring — 16 squares): High-energy zone surrounding Brahma; ideal for family gathering spaces, altars, and quiet verandas.
3. ***Manushyapadam*** (Human Ring — 24 squares): The primary structural ring; ideal for main living rooms, bedrooms, kitchens, and daily human activities.
4. ***Paishachapadam*** (Peripheral Ring — 32 squares): Outer boundary zone; ideal for verandas, setbacks, storage, boundary walls, and external services.

```
+-----------------------------------------------------------------------+
|                       PAISHACHAPADAM (Outer Ring)                     |
|   +---------------------------------------------------------------+   |
|   |                   MANUSHYAPADAM (Human Ring)                  |   |
|   |   +-------------------------------------------------------+   |   |
|   |   |               DAIVIKAPADAM (Divine Ring)              |   |   |
|   |   |   +-----------------------------------------------+   |   |   |
|   |   |   |               BRAHMAPADAM (Core)              |   |   |   |
|   |   |   |            Brahma / Central Courtyard         |   |   |   |
|   |   |   |                                               |   |   |   |
|   |   |   +-----------------------------------------------+   |   |   |
|   |   |                                                       |   |   |
|   |   +-------------------------------------------------------+   |   |
|   |                                                               |   |
|   +---------------------------------------------------------------+   |
|                                                                       |
+-----------------------------------------------------------------------+
```

#### The 32 Peripheral Deities & Auspicious Entrance Doors
The outermost boundary contains 32 presiding deities. Specific deities govern the entry points (*Dwara*); placing main entrance doors over beneficial deity plots guarantees prosperity, while malefic plots cause severe afflictions:

* **Eastern Wall**: *Shikhin* (Fire danger) | *Parjanya* (Excess female births) | **Jayanta** (Immense wealth) ★ | **Mahendra** (Royal/Govt favor) ★ | *Surya* (Extreme wrath) | *Satya* (Falsehood) | *Bhrisha* (Cruelty) | *Antariksha* (Theft).
* **Southern Wall**: *Agni* (Child trouble) | *Pushan* (Slavery) | *Vitatha* (Mean life) | **Brihakshata** (Increase of prosperity & progeny) ★ | *Yama* (Fierceness) | *Gandharva* (Ingratitude) | *Bhringaraja* (Poverty) | *Mriga* (Loss of power).
* **Western Wall**: *Pitri* (Trouble to sons) | *Douvarika* (Increase of enemies) | *Sugriva* (Loss of wealth) | **Pushpadanta** (Abundant prosperity) ★ | **Varuna** (Increase of wealth) ★ | *Asura* (Danger from authority) | *Shosha* (Loss of health) | *Roga* (Ill health).
* **Northern Wall**: *Roga* (Imprisonment) | *Naga* (Enmity) | **Mukhya** (Influx of wealth) ★ | **Soma** (Children & vast wealth) ★ | *Bhallata* (Wealth) | *Aditi* (Wife's flaws) | *Diti* (Poverty) | *Isha* (Misfortune).

---

### 4. SITE SELECTION & EMPIRICAL SOIL DIAGNOSTICS

Before construction, classical Vastu requires rigorous site evaluation using the five senses and physical diagnostics:

#### Site Classification by *Varna* Attributes
Sites are categorized into four classes based on shape, soil color, odor, taste, and natural declivity:

| Varna Class | Plot Shape | Soil Color | Odor / Flora | Taste | Favorable Slope | Attributed Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Brahmin** | Square (\\(1:1\\)) | White | Ghee / Udumbara trees | Sweet | Downward to **North** | Spiritual wisdom, peace, fortune |
| **Kshatriya** | Rectangle (\\(1:1\frac{1}{8}\\)) | Red / Blood | Blood / Ashvattha trees | Bitter / Astringent | Downward to **East** | Power, success, administrative strength |
| **Vaishya** | Rectangle (\\(1:1\frac{1}{6}\\)) | Yellow | Sesame / Plaksha trees | Sour | Downward to **East/South** | Commercial prosperity & wealth |
| **Shudra** | Rectangle (\\(1:1\frac{1}{4}\\)) | Black | Fish / Nyagrodha trees | Pungent | Downward to **West** | Abundant grain, labor yields |

#### Empirical Soil Tests
1. **Compactness Test**: A pit measuring 1 *Hasta* (\\(1\times1\times1\\) cubit) is dug in the center of the site. The excavated earth is thrown back into the pit.
   * *Inferior*: Soil level is below the rim (porous/loose earth).
   * *Average*: Soil level fills the pit exactly to the rim.
   * *Superior*: Soil overflows the pit (dense, compact earth capable of bearing structural loads).
2. **Porosity Test**: The same pit is filled with water at nightfall and examined at dawn.
   * *Inferior*: Water is completely absorbed.
   * *Superior*: Water level remains stable (impermeable clay strata). Clockwise rotation of water indicates supreme bliss.
3. **Oxygenation Test**: An unbaked earthen lamp with four wicks (oriented North, East, South, West) soaked in ghee is lit inside the pit.
   * The wick that burns longest indicates the dominant *Varna* suitabilty; non-lighting indicates anaerobic, unsuitable soil.
4. **Fertility Germination Test**: Seeds (mustard, sesame, barley, wheat) are sown in the pit.
   * *Superior*: Seeds sprout within 3 days.
   * *Inferior*: Seeds take 7 days or fail to sprout.
5. ***Shalya Shodhana*** (Removal of Impurities): Digging the site to clear subsurface "bones, wood, charcoal, and ant hills" before laying foundation bricks (*Prathemestaka*).

---

### 5. REMEDIAL PRINCIPLES & NON-DEMOLITION INTERVENTIONS (*CHIKITSA VAASTU*)

Both Vibhuti Chakrabarti and Sashikala Ananth strongly criticize modern "Vastu consultants" who exploit homeowners through fear-based predictions and drastic demolition. Classical Vastu emphasizes **Chikitsa Vaastu**—the therapeutic correction and energetic healing of existing spaces without structural demolition.

#### 1. Color Therapy (*Panchamahabhuta* Rebalancing)
Zones with defective functional layouts can be rebalanced by introducing elemental colors via wall paints, floor coverings, or curtains:
* **North-East / North (Water)**: Pure **White**, **Blue**, or Light Yellow tones.
* **East / South-East (Fire)**: **Red**, **Coral**, or Pink tones to energize Agni.
* **South / South-West (Earth)**: **Yellow** or **Golden-Brown** tones to ground energy.
* **West / North-West (Air)**: **Grey**, **White**, or Metallic Silver tones.

#### 2. Elemental Neutralization & Material Realignment
* **Metallic Strips / Wire Grounding**: For misplaced toilets or entrances, copper (Fire/East), brass (Earth/South-West), or steel strips are embedded in the floor boundary to block negative energy currents.
* **Neutralizing Steel Reinforcement**: In modern reinforced concrete (RCC) structures, steel cages act as "Faraday cages" that distort natural geomagnetic fields; proper electrical grounding/earthing of the steel structural grid neutralizes ionization stress.
* **Material Swaps**: Replacing high-negative-energy synthetic cladding or polished dark granite (associated with Saturn/depression) with lime plaster, natural sandstone, or marble set in lime mortar.

#### 3. Air Cleansing & Herbal Fumigation (*Sambrani*)
To clear stagnant or negative energies (*Paisaacha*) without altering walls, classical texts (*Mayamata* Ch. 28) prescribe periodic herbal fumigation using resin (*Sambrani*) mixed with:
* *Tulasi* (Holy Basil), *Guggulu* (Resin), *Neem* leaves, *Sarja*, *Mustard* (*Sarshapa*), *Vacha*, and *Sandalwood*.
* This eliminates micro-organisms, repels pests, clears airborne pollutants, and restores positive elemental vibrations.

#### 4. Spatial Realignment & Mirror Counterweights
* **Mirror Extension**: Placing mirrors on North or East walls to visually "extend" cut corners or missing zones.
* **Structural Weighting**: Placing heavy furniture, rock gardens, or storage units in the South-West corner to artificially raise the energy mass and ground the *Pitri* zone.
* **Covering Overhead Beams**: False ceilings or wooden casings installed under exposed load-bearing concrete beams to prevent "beam pressure" (*Vedha*) on occupants seated below.

---

Would you like to generate a structured implementation matrix, a diagnostic checklist for house audits, or specific architectural floor plans based on these principles?
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
