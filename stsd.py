import random
import requests
import re
import json
import os
import shutil
import tempfile
import subprocess
import time
import io
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import yt_dlp

# ============= OPTIONAL IMPORTS (Graceful Fallback) =============
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

try:
    import urllib.request
    from urllib.parse import urljoin
except ImportError:
    pass


# ============= CLOUD-SAFE PATH CONFIGURATION =============
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TEMP_DIR = Path(tempfile.gettempdir())


def app_path(*parts: str) -> str:
    """Build a cloud-safe path relative to the app directory."""
    return str(APP_DIR.joinpath(*parts))


def temp_path(filename: str) -> str:
    """Build a stable temp path for cloud/local runtimes."""
    return str(TEMP_DIR.joinpath(filename))


# ============= yt-dlp COMMAND BUILDER (Cloud-Compatible) =============
def build_yt_dlp_command(url: str, output_template: str, format_selector: str = "best[ext=mp4]/best", section_spec: Optional[str] = None) -> List[str]:
    # قائمة بسيرفرات وسيطة مجانية ومفتوحة المصدر
    invidious_instances = [
        "https://yewtu.be",
        "https://invidious.snopyta.org",
        "https://invidious.kavin.rocks",
        "https://vid.puffyan.us",
        "https://inv.riverside.rocks"
    ]
    
    # تحويل رابط يوتيوب العادي إلى رابط وسيط لتجنب حظر الـ IP
    video_id = extract_video_id(url)
    proxy_url = f"{random.choice(invidious_instances)}/watch?v={video_id}"
    
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720][ext=mp4]", # جودة 720p لضمان السرعة وتجنب الحظر
        "--no-check-certificate",
        "--user-agent", DEFAULT_USER_AGENT,
        "-o", output_template,
        # استخدام العميل المكتبي العادي عبر الوسيط
        "--extractor-args", "youtube:player_client=web",
        "--force-ipv4",
    ]
    
    # إضافة ملف الكوكيز من المسار المباشر داخل المجلد الرئيسي
    cookies_path = app_path("cookies.txt")
    if os.path.exists(cookies_path):
        cmd.extend(["--cookies", cookies_path])
    
    if section_spec:
        cmd.extend(["--download-sections", section_spec, "--force-keyframes-at-cuts"])
    
    cmd.append(proxy_url) # نرسل رابط الوسيط بدلاً من رابط يوتيوب
    return cmd


def tail_text(output_text: str, max_lines: int = 20) -> str:
    """Return the last non-empty lines from a command output string."""
    lines = [line.rstrip() for line in (output_text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def store_yt_dlp_error(stdout_text: str = "", stderr_text: str = "") -> str:
    """Persist yt-dlp diagnostics for the UI and return a compact tail."""
    tail = tail_text("\n".join(part for part in [stdout_text, stderr_text] if part), max_lines=20)
    st.session_state.last_yt_dlp_error = tail
    return tail


# ============= USER-AGENT POOL & COBALT INSTANCE ROTATION =============
USER_AGENTS = [
    DEFAULT_USER_AGENT,
]

def get_random_user_agent() -> str:
    """Return a random user agent from the pool."""
    return random.choice(USER_AGENTS)


COBALT_INSTANCES = [
    "https://cobalt.api.unblocker.it",
    "https://cobalt-api.kwiatekmiki.gq",
    "https://api.cobalt.tools",
    "https://cobalt.api.timothymiller.dev",
    "https://api.cobalt.wtf",
]

def get_cobalt_instances_list() -> List[str]:
    """Fetch and cache list of working Cobalt instances from cobalt.tools."""
    if hasattr(get_cobalt_instances_list, '_cache'):
        return get_cobalt_instances_list._cache
    
    try:
        response = requests.get(
            "https://cobalt.tools/api/instances",
            timeout=10,
            headers={"User-Agent": get_random_user_agent()}
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                instances = [inst.get("api") for inst in data if isinstance(inst, dict) and inst.get("api")]
                instances = [inst for inst in instances if inst and inst.startswith("http")]
                if instances:
                    get_cobalt_instances_list._cache = instances
                    return instances
    except Exception:
        pass
    
    # Fallback to hardcoded instances if fetching fails
    get_cobalt_instances_list._cache = COBALT_INSTANCES
    return COBALT_INSTANCES


def get_rotating_cobalt_instance() -> str:
    """Get a Cobalt instance with fallback rotation."""
    instances = get_cobalt_instances_list()
    return random.choice(instances) if instances else "https://api.cobalt.tools"


# ============= TRANSCRIPTION INTELLIGENCE ENGINE (Optimized) =============
# Replaced local whisper-based transcription block with a lightweight
# cloud-smart transcript fetcher that uses only youtube_transcript_api.
def _normalize_transcript_segments(raw_segments: List[Dict]) -> List[Dict]:
    """Normalize transcript segments into the app's expected structure."""
    normalized = []
    for index, segment in enumerate(raw_segments or []):
        if isinstance(segment, dict):
            text_value = segment.get("text", "")
            start_raw = segment.get("start", index * 5.0)
            duration_raw = segment.get("duration", 5.0)
        else:
            text_value = getattr(segment, "text", "")
            start_raw = getattr(segment, "start", index * 5.0)
            duration_raw = getattr(segment, "duration", 5.0)
        text = _normalize_transcript_text(str(text_value))
        if not text:
            continue
        start_value = float(start_raw or index * 5.0)
        duration_value = float(duration_raw or 5.0)
        normalized.append({
            "text": text,
            "start": start_value,
            "duration": duration_value,
        })
    return normalized


def _normalize_transcript_text(text: str) -> str:
    """Keep raw transcript text and only normalize whitespace/tags safely."""
    cleaned_text = re.sub(r"<[^>]+>", " ", text or "")
    cleaned_text = cleaned_text.replace("\r", " ").replace("\n", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def _dedupe_repeated_words(text: str) -> str:
    """Remove immediate repeated words like 'go go go' -> 'go'."""
    words = (text or "").split()
    if not words:
        return ""
    output_words = [words[0]]
    for word in words[1:]:
        if word.lower() != output_words[-1].lower():
            output_words.append(word)
    return " ".join(output_words)


def _clean_transcript_for_analysis(transcript: List[Dict]) -> List[Dict]:
    """Remove non-informative caption labels and repeated noise before AI analysis."""
    cleaned_segments: List[Dict] = []
    noise_pattern = re.compile(r"^\[(music|applause|laughter|noise|silence)\]$", re.IGNORECASE)
    for seg in transcript or []:
        raw_text = _normalize_transcript_text(str(seg.get("text", "")))
        raw_text = re.sub(r"\[(music|applause|laughter|noise|silence)\]", " ", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\((music|applause|laughter|noise|silence)\)", " ", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        if not raw_text or noise_pattern.match(raw_text):
            continue
        clean_text = _dedupe_repeated_words(raw_text)
        if not clean_text:
            continue
        if cleaned_segments and cleaned_segments[-1].get("text", "").strip().lower() == clean_text.lower():
            continue
        cleaned_segments.append({
            "text": clean_text,
            "start": float(seg.get("start", 0.0) or 0.0),
            "duration": float(seg.get("duration", 5.0) or 5.0),
        })
    return cleaned_segments


def _parse_vtt_to_segments(vtt_text: str) -> List[Dict]:
    """Parse VTT subtitle text into transcript segments."""
    def parse_vtt_timestamp(ts: str) -> float:
        ts = ts.replace(",", ".").strip()
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])

    transcript = []
    current_start = 0.0
    current_end = 5.0

    for raw_line in (vtt_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        if "-->" in line:
            try:
                start_str, end_str = [part.strip() for part in line.split("-->", 1)]
                current_start = parse_vtt_timestamp(start_str.split(" ")[0])
                current_end = parse_vtt_timestamp(end_str.split(" ")[0])
            except Exception:
                current_end = current_start + 5.0
            continue

        if line.isdigit():
            continue

        clean_text = _normalize_transcript_text(line)
        if not clean_text:
            continue

        transcript.append({
            "text": clean_text,
            "start": current_start,
            "duration": max(0.5, current_end - current_start),
        })

    return transcript


def _parse_json3_to_segments(json3_text: str) -> List[Dict]:
    """Parse yt-dlp JSON3 subtitle text into transcript segments."""
    try:
        data = json.loads(json3_text)
    except Exception:
        return []

    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        return []

    transcript = []
    for event in events:
        if not isinstance(event, dict):
            continue
        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs") or event.get("durMs") or event.get("tDurationMs")
        segments = event.get("segs") or []
        if start_ms is None or not isinstance(segments, list):
            continue

        text_parts = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text_value = _normalize_transcript_text(str(segment.get("utf8", "")))
            if text_value:
                text_parts.append(text_value)

        text = " ".join(text_parts).strip()
        if not text:
            continue

        transcript.append({
            "text": text,
            "start": float(start_ms) / 1000.0,
            "duration": max(0.5, float(duration_ms or 5000) / 1000.0),
        })

    return transcript


def _fetch_transcript_with_ytdlp(video_id: str) -> Optional[List[Dict]]:
    """Fetch raw subtitles with yt-dlp while preferring original English when available."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = temp_path(f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-format", "json3/srt/vtt",
        "--sub-langs", "en.*,ar.*",
        "--no-check-certificate",
        "--prefer-insecure",
        "--user-agent", DEFAULT_USER_AGENT,
        "--referer", "https://www.youtube.com/",
        "-o", output_template,
        video_url,
    ]
    cookies_path = app_path("cookies.txt")
    if os.path.exists(cookies_path):
        cmd[1:1] = ["--cookies", cookies_path]

    expected_files = []
    for ext in ("vtt", "json", "json3", "srt"):
        expected_files.extend(TEMP_DIR.glob(f"{video_id}*.{ext}"))

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None

    subtitle_files = sorted({*expected_files, *TEMP_DIR.glob(f"{video_id}*.vtt"), *TEMP_DIR.glob(f"{video_id}*.json"), *TEMP_DIR.glob(f"{video_id}*.json3"), *TEMP_DIR.glob(f"{video_id}*.srt")})
    if not subtitle_files:
        return None

    def subtitle_priority(file_path: Path) -> int:
        lower_name = file_path.name.lower()
        # Prefer English original tracks first, then Arabic, then others.
        if ".en-orig." in lower_name or ".en.orig." in lower_name:
            return 0
        if ".en." in lower_name or ".en-" in lower_name:
            return 1
        if ".ar-orig." in lower_name or ".ar.orig." in lower_name:
            return 2
        if ".ar." in lower_name or ".ar-" in lower_name:
            return 3
        return 9

    selected_file = sorted(subtitle_files, key=subtitle_priority)[0]

    try:
        content = selected_file.read_text(encoding="utf-8", errors="ignore")
        if selected_file.suffix.lower() in (".json", ".json3"):
            transcript = _parse_json3_to_segments(content)
        else:
            transcript = _parse_vtt_to_segments(content)
        transcript = [
            {
                "text": _normalize_transcript_text(segment.get("text", "")),
                "start": float(segment.get("start", 0.0) or 0.0),
                "duration": float(segment.get("duration", 5.0) or 5.0),
            }
            for segment in transcript
            if _normalize_transcript_text(segment.get("text", ""))
        ]
        return transcript if transcript else None
    except Exception:
        return None
    finally:
        for file_path in subtitle_files:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass


def _fetch_transcript_from_proxy(video_id: str) -> Optional[List[Dict]]:
    """Try lightweight proxy caption endpoints when direct YouTube transcript lookup fails."""
    proxy_endpoints = [
        f"https://invidious.snopyta.org/api/v1/captions/{video_id}",
        f"https://yewtu.be/api/v1/captions/{video_id}",
        f"https://vid.puffyan.us/api/v1/captions/{video_id}",
        f"https://inv.nadeko.net/api/v1/captions/{video_id}",
        f"https://piped.video/api/v1/captions/{video_id}",
        f"https://piped.kavin.rocks/api/v1/captions/{video_id}",
        f"https://piped-mirror.kavin.rocks/api/v1/captions/{video_id}",
    ]

    def parse_caption_payload(payload: Dict) -> Optional[List[Dict]]:
        captions = payload.get("captions") or payload.get("subtitles") or []
        if not isinstance(captions, list) or not captions:
            return None

        chosen_caption = None
        # Prefer original English captions, then Arabic, then fallback.
        for caption in captions:
            language_code = str(
                caption.get("language_code")
                or caption.get("code")
                or caption.get("lang")
                or caption.get("srclang")
                or ""
            ).lower()
            if language_code.startswith("en"):
                chosen_caption = caption
                break
        if chosen_caption is None:
            for caption in captions:
                language_code = str(
                    caption.get("language_code")
                    or caption.get("code")
                    or caption.get("lang")
                    or caption.get("srclang")
                    or ""
                ).lower()
                if language_code.startswith("ar"):
                    chosen_caption = caption
                    break

        if chosen_caption is None:
            chosen_caption = captions[0]

        caption_url = chosen_caption.get("url") or chosen_caption.get("uri") or chosen_caption.get("baseUrl")
        if not caption_url:
            return None

        if caption_url.startswith("/"):
            base_url = "https://" + endpoint.split("/")[2]
            caption_url = f"{base_url}{caption_url}"

        try:
            caption_response = requests.get(caption_url, timeout=10, headers={"User-Agent": DEFAULT_USER_AGENT})
        except Exception:
            return None

        if caption_response.status_code != 200:
            return None

        content_type = caption_response.headers.get("content-type", "").lower()
        caption_text = caption_response.text.strip()
        if not caption_text:
            return None

        # Some invidious servers may return JSON caption chunks instead of VTT.
        if "application/json" in content_type or caption_text.startswith("["):
            try:
                json_items = caption_response.json()
                return _normalize_transcript_segments(json_items)
            except Exception:
                pass

        transcript = []
        current_time = 0.0
        for line in caption_text.splitlines():
            cleaned_line = line.strip()
            if not cleaned_line or cleaned_line.startswith(("WEBVTT", "NOTE")):
                continue
            if "-->" in cleaned_line:
                start_str = cleaned_line.split("-->")[0].strip()
                parts = start_str.split(":")
                try:
                    if len(parts) == 3:
                        current_time = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                except Exception:
                    pass
                continue
            if len(cleaned_line) > 1 and "[" not in cleaned_line:
                transcript.append({
                    "text": cleaned_line,
                    "start": current_time,
                    "duration": 5.0,
                })
                current_time += 5.0

        return transcript if transcript else None

    for endpoint in proxy_endpoints:
        try:
            response = requests.get(endpoint, timeout=10, headers={"User-Agent": DEFAULT_USER_AGENT})
            if response.status_code != 200:
                continue
            payload = response.json()
            transcript = parse_caption_payload(payload)
            if transcript:
                return transcript
        except Exception:
            continue

    return None


def _get_transcript_smart_result(video_id: str) -> Tuple[Optional[List[Dict]], str]:
    """
    Lightweight cloud-based transcript fetcher.

    Strict order:
    1) Manual transcripts (en/ar) via YouTubeTranscriptApi
    2) Auto subtitles via yt-dlp (--write-auto-subs)
    3) Proxy captions endpoints
    4) None (description is handled outside as last resort)
    """
    manual_or_generated = fetch_transcript(video_id, languages=["en", "ar"])
    if manual_or_generated:
        normalized = _clean_transcript_for_analysis(_normalize_transcript_segments(manual_or_generated))
        if normalized:
            return normalized, "youtube_transcript_api"

    ytdlp_transcript = _fetch_transcript_with_ytdlp(video_id)
    if ytdlp_transcript:
        return _clean_transcript_for_analysis(ytdlp_transcript), "ytdlp_subtitles"

    proxy_transcript = _fetch_transcript_from_proxy(video_id)
    if proxy_transcript:
        return _clean_transcript_for_analysis(proxy_transcript), "proxy_captions"

    return None, "none"


def get_transcript_smart(video_id: str) -> Optional[List[Dict]]:
    transcript, _source = _get_transcript_smart_result(video_id)
    if not transcript:
        return None
    cleaned = _clean_transcript_for_analysis(transcript)
    return [
        {
            "text": _normalize_transcript_text(segment.get("text", "")),
            "start": float(segment.get("start", 0.0) or 0.0),
            "duration": float(segment.get("duration", 5.0) or 5.0),
        }
        for segment in cleaned
        if _normalize_transcript_text(segment.get("text", ""))
    ]


# ============= PAGE CONFIG & STYLING =============
st.set_page_config(
    page_title="ViraFlow - Viral Shorts Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional luxury theme with animations
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0b0c;
    color: #efe7d6;
}
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stApp {
    background-color: #0b0b0c;
}

/* Header Branding */
.header-title {
    background: linear-gradient(135deg, #ffd700 0%, #b8860b 50%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: 800;
    text-align: center;
    padding: 30px 0;
    letter-spacing: 2px;
}

/* Stage Badge */
.stage-badge {
    display: inline-block;
    background: linear-gradient(90deg, #b8860b, #ffd700);
    color: #0b0b0c;
    padding: 10px 20px;
    border-radius: 20px;
    font-weight: 600;
    margin: 15px 0;
    font-size: 1.1em;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(135deg, #b8860b 0%, #ffd700 100%);
    color: #0b0b0c;
    font-weight: 700;
    border: none;
    padding: 12px 24px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #ffd700 0%, #b8860b 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4);
}

/* Transcript & Viral Cards */
.transcript-box {
    background-color: #1a1a1d;
    border-left: 4px solid #ffd700;
    padding: 15px;
    border-radius: 5px;
    font-family: 'Courier New', monospace;
    max-height: 500px;
    overflow-y: auto;
    font-size: 0.9em;
    line-height: 1.6;
}

.viral-moment-card {
    background: linear-gradient(135deg, rgba(184, 134, 11, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%);
    border: 1px solid rgba(255, 215, 0, 0.3);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
}
.viral-moment-card:hover {
    border-color: #ffd700;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
}

/* Metrics */
.metric-value {
    color: #ffd700;
    font-size: 2.2em;
    font-weight: 700;
    text-align: center;
}

/* Status Messages */
.status-text {
    color: #ffd700;
    font-family: 'Courier New', monospace;
    font-size: 0.95em;
    font-weight: 600;
}

/* Two-Column Layout */
.two-column-layout {
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 20px;
    margin: 20px 0;
}

@media (max-width: 1200px) {
    .two-column-layout {
        grid-template-columns: 1fr;
    }
}

/* Footer */
.footer-branding {
    text-align: center;
    padding: 20px;
    border-top: 2px solid rgba(198, 184, 154, 0.3);
    margin-top: 40px;
    color: #c7b89a;
    font-size: 0.9em;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ============= SESSION STATE INITIALIZATION =============
def init_session_state():
    """Initialize all session state variables for production workflow."""
    defaults = {
        "stage": 1,  # 1=Input, 2=Selection, 3=Rendering
        "video_id": None,
        "url": None,
        "transcript": None,
        "transcript_text": None,
        "viral_moments": [],
        "selected_moment": None,
        "output_video": None,
        "quality": "720p",
        "aspect_ratio": "9:16",
        "history": [],
        "languages": ["en"],
        "transcript_source": None,
        "custom_keywords": "",
        "use_custom_keywords": False,
        "transcription_status": None,
        "last_yt_dlp_error": "",
        "download_link": None,  # Direct Cobalt download URL
        "download_triggered": False,  # Auto-download trigger flag
        "fast_mode_enabled": False,  # Fast Engine mode flag
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ============= PROFESSIONAL HEADER & FOOTER =============
st.markdown('<div class="header-title">🎬 ViraFlow | Viral Shorts Engine</div>', unsafe_allow_html=True)
st.markdown("---")


# ============= UTILITY FUNCTIONS =============
def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract YouTube video ID from URL or validate if already an ID."""
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?[^\s]*v=([A-Za-z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


@st.cache_data
def fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> Optional[List[Dict]]:
    """
    Fetch transcript (both captions and auto-generated) from YouTube.
    
    IMPORTANT DISTINCTION:
    - Captions: Manually created subtitles by uploader (rare)
    - Transcripts: Auto-generated from speech-to-text (very common!)
    
    This function fetches BOTH types.
    """
    try:
        cookies_path = app_path("cookies.txt")
        
        # Initialize API with cookies if available
        if os.path.exists(cookies_path):
            try:
                api = YouTubeTranscriptApi(cookies=cookies_path)
            except TypeError:
                try:
                    api = YouTubeTranscriptApi(cookie_path=cookies_path)
                except TypeError:
                    api = YouTubeTranscriptApi()
        else:
            api = YouTubeTranscriptApi()

        # Strategy 1: List all available transcripts
        try:
            transcripts = api.list_transcripts(video_id)
        except (TranscriptsDisabled, VideoUnavailable):
            return None
        except Exception:
            return None
        
        # Build language preferences
        preferred_langs = languages if languages else ["en", "ar", "es", "fr", "pt", "de", "ja", "ru", "zh-Hans"]
        
        # Strategy 2: Try manually created transcripts first (higher quality)
        for lang in preferred_langs:
            for transcript in transcripts.manually_created_transcripts:
                if transcript.language_code.startswith(lang):
                    try:
                        return transcript.fetch()
                    except Exception:
                        pass
        
        # Strategy 3: Fall back to auto-generated transcripts (almost always available!)
        for lang in preferred_langs:
            for transcript in transcripts.generated_transcripts:
                if transcript.language_code.startswith(lang):
                    try:
                        return transcript.fetch()
                    except Exception:
                        pass
        
        # Strategy 4: If preferred languages not found, use first manually created (any language)
        if transcripts.manually_created_transcripts:
            try:
                return transcripts.manually_created_transcripts[0].fetch()
            except Exception:
                pass
        
        # Strategy 5: Last resort - use first auto-generated transcript (ANY language)
        if transcripts.generated_transcripts:
            try:
                return transcripts.generated_transcripts[0].fetch()
            except Exception:
                pass
        
        return None
    except Exception:
        return None


def format_transcript(transcript: List[Dict], show_timestamps: bool = True) -> str:
    """Format transcript with optional timestamps."""
    lines = []
    for seg in transcript:
        text = seg.get("text", "").strip()
        if show_timestamps:
            start = int(seg.get("start", 0))
            mm = start // 60
            ss = start % 60
            ts = f"[{mm:02d}:{ss:02d}] "
        else:
            ts = ""
        lines.append(f"{ts}{text}")
    return "\n".join(lines)


def compute_viral_score(text: str, keywords: List[str]) -> float:
    """
    Compute viral score for a transcript segment.
    Weights high-impact emotional words and user keywords.
    """
    score = 0.0
    
    high_impact = ["shocking", "incredible", "amazing", "mind-blowing", "unbelievable", 
                   "breaking", "viral", "shocking", "insane", "legendary", "epic"]
    medium_impact = ["secret", "hidden", "truth", "exclusive", "revealed", "wow"]
    low_impact = ["money", "system", "trick", "hack", "tip", "fast", "easy"]
    
    text_lower = text.lower()
    score += sum(3.0 for kw in high_impact if kw in text_lower)
    score += sum(2.0 for kw in medium_impact if kw in text_lower)
    score += sum(1.0 for kw in low_impact if kw in text_lower)
    score += sum(1.5 for kw in keywords if kw.lower() in text_lower)
    score += 1.0 if "!" in text else 0.0
    score += 0.5 if "?" in text else 0.0
    score += 0.5 if any(word.isupper() and len(word) > 3 for word in text.split()) else 0.0
    
    return min(100.0, score * 10)


def find_viral_moments(
    transcript: List[Dict],
    custom_keywords: Optional[List[str]] = None,
    top_n: int = 3
) -> List[Dict]:
    """
    AI-powered viral moment detection with custom keyword support.
    """
    if not transcript or len(transcript) < 2:
        return []
    
    default_keywords = [
        "shocking", "incredible", "amazing", "secret", "truth",
        "exclusive", "money", "hack", "system", "trick"
    ]
    
    keywords = default_keywords + (custom_keywords or [])
    moments = []
    
    for i in range(len(transcript) - 3):
        window = " ".join([seg.get("text", "") for seg in transcript[i:i+4]])
        if len(window.strip()) < 20:
            continue
        
        start_time = float(transcript[i].get("start", 0))
        viral_score = compute_viral_score(window, keywords)
        
        if viral_score > 10:
            top_keyword = "Viral Clip"
            for kw in keywords:
                if kw.lower() in window.lower():
                    top_keyword = kw.capitalize()
                    break
            
            end_time = start_time + 45
            mm = int(start_time) // 60
            ss = int(start_time) % 60
            ts = f"{mm:02d}:{ss:02d}"
            
            moments.append({
                "title": top_keyword,
                "timestamp": ts,
                "start_time": start_time,
                "end_time": end_time,
                "viral_score": min(100, viral_score),
                "snippet": window[:100] + "..." if len(window) > 100 else window,
            })
    
    moments.sort(key=lambda x: x["viral_score"], reverse=True)
    return moments[:top_n]


def _repunctuate_transcript(transcript: List[Dict]) -> List[Dict]:
    """Break dense transcript blocks into cleaner sentence-like segments before AI scoring."""
    if not transcript:
        return []

    merged_text = " ".join(_normalize_transcript_text(seg.get("text", "")) for seg in transcript if _normalize_transcript_text(seg.get("text", "")))
    if not merged_text:
        return []

    merged_text = re.sub(r"\s+", " ", merged_text).strip()
    if len(merged_text) < 120:
        return [{
            "text": merged_text.strip(),
            "start": float(transcript[0].get("start", 0.0) or 0.0),
            "duration": float(transcript[0].get("duration", 5.0) or 5.0),
        }]

    sentence_candidates = re.split(r"(?<=[.!?])\s+|(?<=\))\s+", merged_text)
    sentence_candidates = [sentence.strip() for sentence in sentence_candidates if sentence and sentence.strip()]

    if not sentence_candidates:
        sentence_candidates = [merged_text]

    repunctuated = []
    start_time = float(transcript[0].get("start", 0.0) or 0.0)
    for index, sentence in enumerate(sentence_candidates):
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue
        repunctuated.append({
            "text": cleaned_sentence,
            "start": start_time + index * 5.0,
            "duration": 5.0,
        })

    return repunctuated if repunctuated else transcript


def analyze_with_ai(transcript: List[Dict], custom_keywords: Optional[List[str]] = None, top_n: int = 3) -> List[Dict]:
    """Primary AI analysis entry point for transcript-driven viral moment detection."""
    if not transcript:
        return []
    transcript = _clean_transcript_for_analysis(transcript)
    if not transcript:
        return []

    def _looks_english_text(text: str) -> bool:
        letters = re.findall(r"[A-Za-z]", text or "")
        if not letters:
            return False
        non_space_chars = re.findall(r"\S", text or "")
        return len(letters) >= max(20, int(len(non_space_chars) * 0.45))

    def _analyze_with_openai(raw_transcript: List[Dict]) -> Optional[List[Dict]]:
        if not HAS_OPENAI:
            return None

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        transcript_text = format_transcript(raw_transcript, show_timestamps=True)
        is_english = _looks_english_text(transcript_text)
        keyword_text = ", ".join(custom_keywords or [])

        system_prompt = (
            "You are an expert short-form video editor and viral moments analyst. "
            "Always analyze the transcript in its original language. Never translate or rewrite the transcript content."
        )
        user_prompt = f"""
حلل النص التالي بلغته الأصلية. استخرج أفضل {top_n} لقطات مشوقة.
أرجع النتيجة بتنسيق JSON يحتوي على Start_Time و Duration و Title.

Constraints:
- Keep the transcript content as-is (raw), no translation.
- Use timestamps grounded in the provided [MM:SS] transcript lines.
- Return JSON array only (no markdown, no explanations).
- If the transcript is English, make Title bilingual (Arabic | English).
- If custom keywords exist, prioritize them: {keyword_text if keyword_text else "none"}.
- حتى لو كان النص غير واضح تماماً، حاول استنتاج أفضل 3 لقطات بناءً على كثافة الكلمات أو تغير المواضيع. لا تعِد نتيجة فارغة أبداً.

Expected JSON item shape:
{{
  "Start_Time": <number_seconds>,
  "Duration": <number_seconds>,
  "Title": "<string>"
}}

Detected English transcript: {"yes" if is_english else "no"}

RAW TRANSCRIPT:
{transcript_text}
"""
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_content = (response.choices[0].message.content or "").strip()
            if not raw_content:
                return None

            cleaned_json = raw_content
            fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
            if fenced_match:
                cleaned_json = fenced_match.group(1).strip()

            parsed = json.loads(cleaned_json)
            if not isinstance(parsed, list):
                return None

            normalized: List[Dict] = []
            for item in parsed[:top_n]:
                if not isinstance(item, dict):
                    continue
                title_raw = str(
                    item.get("Title")
                    or item.get("title")
                    or item.get("TITLE")
                    or ""
                ).strip()
                start_time = float(
                    item.get("Start_Time")
                    or item.get("start_time")
                    or item.get("start")
                    or 0.0
                )
                duration = float(
                    item.get("Duration")
                    or item.get("duration")
                    or 45.0
                )
                end_time = start_time + max(1.0, duration)
                if end_time <= start_time:
                    end_time = start_time + 45.0
                mm = int(start_time) // 60
                ss = int(start_time) % 60
                timestamp = f"{mm:02d}:{ss:02d}"
                snippet = ""
                normalized.append({
                    "title": title_raw if title_raw else "لقطة مشوقة | Viral Moment",
                    "timestamp": timestamp,
                    "start_time": start_time,
                    "end_time": end_time,
                    "viral_score": 90.0,
                    "snippet": snippet,
                })

            if normalized:
                while len(normalized) < top_n:
                    idx = len(normalized)
                    start_time = float(transcript[min(idx, len(transcript) - 1)].get("start", idx * 15.0))
                    mm = int(start_time) // 60
                    ss = int(start_time) % 60
                    normalized.append({
                        "title": f"لقطة مقترحة {idx + 1} | Suggested Moment {idx + 1}",
                        "timestamp": f"{mm:02d}:{ss:02d}",
                        "start_time": start_time,
                        "end_time": start_time + 45.0,
                        "viral_score": 75.0,
                        "snippet": transcript[min(idx, len(transcript) - 1)].get("text", ""),
                    })
            return normalized[:top_n] if normalized else None
        except Exception:
            return None

    ai_moments = _analyze_with_openai(transcript)
    if ai_moments:
        return ai_moments

    repunctuated_transcript = _repunctuate_transcript(transcript)
    fallback = find_viral_moments(repunctuated_transcript, custom_keywords=custom_keywords, top_n=top_n)
    if not fallback:
        for idx in range(min(top_n, len(transcript))):
            start_time = float(transcript[idx].get("start", idx * 15.0))
            mm = int(start_time) // 60
            ss = int(start_time) % 60
            fallback.append({
                "title": f"لقطة مقترحة {idx + 1} | Suggested Moment {idx + 1}",
                "timestamp": f"{mm:02d}:{ss:02d}",
                "start_time": start_time,
                "end_time": start_time + 45.0,
                "viral_score": 70.0,
                "snippet": transcript[idx].get("text", ""),
            })
    # Keep bilingual titles in fallback path when transcript is mostly English.
    if _looks_english_text(format_transcript(transcript, show_timestamps=False)):
        for moment in fallback:
            en_title = str(moment.get("title", "Viral Clip")).strip()
            moment["title_en"] = en_title
            moment["title_ar"] = "لقطة فيروسية"
            moment["title"] = f"{moment['title_ar']} | {moment['title_en']}"
    return fallback


def format_hhmmss(seconds_value: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(float(seconds_value))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_fast_engine_metadata(video_id: str) -> Optional[Dict]:
    """
    FAST ENGINE: Quick metadata fetch when transcript is slow.
    Returns video info and quick viral analysis without full transcripts.
    Timeout: 10 seconds max.
    """
    try:
        # Try to get video info quickly
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 10,
            'default_search': 'auto',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if info:
                return {
                    "title": info.get("title", ""),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", ""),
                    "description": info.get("description", "")[:500],  # First 500 chars
                }
    except Exception:
        pass
    return None


def generate_viral_moments_from_description(description: str, title: str) -> List[Dict]:
    """
    FAST ENGINE: Generate quick viral moments from video metadata when transcripts fail.
    Analyzes title and description to create 3 default moments.
    """
    moments = []
    
    # Check for viral keywords in title/description
    text = f"{title} {description}".lower()
    keywords_found = []
    
    viral_keywords = {
        "shocking": 95,
        "incredible": 85,
        "amazing": 80,
        "secret": 75,
        "hack": 70,
        "money": 65,
        "trending": 90,
        "viral": 100,
        "exclusive": 75,
    }
    
    for keyword, score in viral_keywords.items():
        if keyword in text:
            keywords_found.append((keyword, score))
    
    if keywords_found:
        keywords_found.sort(key=lambda x: x[1], reverse=True)
        
        # Create 3 moments based on found keywords
        for idx, (keyword, score) in enumerate(keywords_found[:3]):
            start_time = idx * 45  # Stagger moments
            mm = start_time // 60
            ss = start_time % 60
            
            moments.append({
                "title": keyword.capitalize(),
                "timestamp": f"{mm:02d}:{ss:02d}",
                "start_time": float(start_time),
                "end_time": float(start_time + 45),
                "viral_score": min(100, score),
                "snippet": description[:80] + "..." if description else "Fast Engine moment",
            })
    else:
        # Default moments if no keywords found
        for i in range(3):
            moments.append({
                "title": f"Moment {i+1}",
                "timestamp": f"00:{i*15:02d}",
                "start_time": float(i * 15),
                "end_time": float(i * 15 + 45),
                "viral_score": 50,
                "snippet": "استخدام المحرك السريع",
            })
    
    return moments


# ============= VERTICAL VIDEO RENDERING (9:16 Mastering) =============
def create_html_download_trigger(download_url: str, filename: str = "video.mp4") -> str:
    """
    Create an HTML/JavaScript snippet that auto-triggers browser download.
    This opens the download dialog immediately without page redirect.
    """
    # Escape URL for JavaScript
    safe_url = download_url.replace('"', '\\"').replace("'", "\\'")
    
    html_code = f"""
    <script>
    (function() {{
        // Create invisible link element
        const link = document.createElement('a');
        link.href = "{safe_url}";
        link.download = "{filename}";
        link.style.display = 'none';
        
        // Append to DOM and trigger click
        document.body.appendChild(link);
        link.click();
        
        // Clean up
        document.body.removeChild(link);
        
        // Show status
        console.log('Download triggered for: {filename}');
    }})();
    </script>
    
    <div style="text-align: center; padding: 20px; background: #0b0b0c; border-radius: 10px; border: 2px solid #ffd700;">
        <p style="color: #ffd700; font-weight: bold; font-size: 16px;">
            ⬇️ التحميل جاري من جهازك الآن...
        </p>
        <p style="color: #c7b89a; font-size: 14px;">
            إذا لم يبدأ التحميل، اضغط <a href="{safe_url}" download="{filename}" style="color: #ffd700; text-decoration: underline;">هنا</a>
        </p>
    </div>
    """
    return html_code


def create_browser_cobalt_redirect_html(video_url: str, start_time: float, end_time: float) -> str:
    """
    Build a browser-side Cobalt launcher.
    No server-side download: the user's browser opens cobalt.tools directly.
    """
    safe_video_url = (video_url or "").replace('"', '\\"').replace("'", "\\'")
    start_seconds = max(0, int(float(start_time or 0)))
    end_seconds = max(start_seconds + 1, int(float(end_time or (start_seconds + 45))))
    cobalt_page_url = f"https://cobalt.tools/?u={safe_video_url}"
    return f"""
    <div style="text-align:center;padding:16px;border:1px solid rgba(255,215,0,0.3);border-radius:10px;background:#111;">
      <button id="open-cobalt-btn" style="background:#ffd700;color:#0b0b0c;border:none;padding:12px 20px;border-radius:8px;font-weight:700;cursor:pointer;">
        📥 تحميل عبر متصفحك (Cobalt)
      </button>
      <p style="color:#c7b89a;margin-top:10px;font-size:13px;">
        سيُفتح Cobalt في تبويب جديد. ابدأ القص من {start_seconds}s إلى {end_seconds}s.
      </p>
    </div>
    <script>
      (function() {{
        const btn = document.getElementById('open-cobalt-btn');
        if (!btn) return;
        btn.addEventListener('click', function() {{
          window.open("{cobalt_page_url}", "_blank", "noopener,noreferrer");
        }});
      }})();
    </script>
    """


def get_cobalt_direct_download_link(
    video_url: str,
    quality_level: int,
    status_placeholder = None,
) -> Tuple[bool, str]:
    """
    ULTRA-FAST PARADIGM: Get direct download link from Cobalt instantly.
    No server processing, no waiting. Just get URL and return.
    """
    
    payload = {
        "url": video_url,
        "videoQuality": quality_level,
        "downloadMode": "auto",
        "filenameStyle": "nerdy",
        "isNoAudio": False,
    }

    # Try multiple Cobalt instances with random User-Agents
    instances = get_cobalt_instances_list()
    random.shuffle(instances)
    
    for instance in instances:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": get_random_user_agent(),
            "Referer": "https://cobalt.tools/",
        }

        try:
            api_response = requests.post(
                f"{instance}/api/json",
                json=payload,
                headers=headers,
                timeout=20,  # Reduced timeout for speed
            )
        except requests.RequestException:
            continue

        if api_response.status_code >= 400:
            continue

        try:
            api_data = api_response.json()
        except Exception:
            continue

        # Extract direct download URL from response
        def extract_stream_url(api_data: Any) -> Optional[str]:
            if isinstance(api_data, dict):
                for key in ("url", "download", "stream", "streamUrl", "link"):
                    value = api_data.get(key)
                    if isinstance(value, str) and value.startswith("http"):
                        return value
                nested_data = api_data.get("data")
                nested_url = extract_stream_url(nested_data)
                if nested_url:
                    return nested_url
                links = api_data.get("links")
                if isinstance(links, list):
                    for item in links:
                        nested_url = extract_stream_url(item)
                        if nested_url:
                            return nested_url
                files = api_data.get("files")
                if isinstance(files, list):
                    for item in files:
                        nested_url = extract_stream_url(item)
                        if nested_url:
                            return nested_url
            elif isinstance(api_data, list):
                for item in api_data:
                    nested_url = extract_stream_url(item)
                    if nested_url:
                        return nested_url
            return None

        stream_url = extract_stream_url(api_data)
        if stream_url:
            return True, stream_url
    
    return False, "لم نتمكن من الاتصال بـ Cobalt. حاول مرة أخرى."


def download_with_cobalt(
    video_url: str,
    start_time: float,
    end_time: float,
    temp_clip_path: str,
    quality_level: int,
    status_placeholder = None,
) -> Tuple[bool, str]:
    """Download a clipped segment through the current Cobalt API."""
    if os.path.exists(temp_clip_path):
        try:
            os.remove(temp_clip_path)
        except Exception:
            pass

    payload = {
        "url": video_url,
        "videoQuality": quality_level,
        "downloadMode": "auto",
        "filenameStyle": "nerdy",
        "isNoAudio": False,
        "sectionStart": round(max(0.0, float(start_time)), 3),
        "sectionEnd": round(max(0.0, float(end_time)), 3),
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }

    if status_placeholder:
        status_placeholder.info(f"🛡️ Downloading clipped segment via Cobalt (quality {quality_level})...")

    try:
        api_response = requests.post(
            "https://api.cobalt.tools/api/json",
            json=payload,
            headers=headers,
            timeout=45,
        )
    except requests.RequestException as e:
        return False, f"Cobalt API request failed: {e}"

    if api_response.status_code >= 400:
        return False, f"Cobalt API HTTP {api_response.status_code}: {tail_text(api_response.text)}"

    try:
        api_data = api_response.json()
    except Exception as e:
        return False, f"Invalid JSON from Cobalt API: {e}"

    def extract_stream_url(api_data: Any) -> Optional[str]:
        if isinstance(api_data, dict):
            for key in ("url", "download", "stream", "streamUrl"):
                value = api_data.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value

            nested_data = api_data.get("data")
            nested_url = extract_stream_url(nested_data)
            if nested_url:
                return nested_url

            links = api_data.get("links")
            if isinstance(links, list):
                for item in links:
                    nested_url = extract_stream_url(item)
                    if nested_url:
                        return nested_url

            files = api_data.get("files")
            if isinstance(files, list):
                for item in files:
                    nested_url = extract_stream_url(item)
                    if nested_url:
                        return nested_url
        elif isinstance(api_data, list):
            for item in api_data:
                nested_url = extract_stream_url(item)
                if nested_url:
                    return nested_url
        return None

    stream_url = extract_stream_url(api_data)
    if not stream_url:
        return False, f"Cobalt API did not return a stream URL: {tail_text(json.dumps(api_data, ensure_ascii=False), max_lines=5)}"

    try:
        with requests.get(stream_url, stream=True, timeout=120) as download_response:
            download_response.raise_for_status()
            with open(temp_clip_path, "wb") as out_file:
                for chunk in download_response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        out_file.write(chunk)
    except requests.RequestException as e:
        return False, f"Failed to download stream from Cobalt URL: {e}"
    except Exception as e:
        return False, f"Failed saving clipped video: {e}"

    if os.path.exists(temp_clip_path) and os.path.getsize(temp_clip_path) > 100000:
        return True, ""

    return False, "Cobalt finished but produced an empty/too-small clip."


def render_short_clip_ffmpeg(
    input_file: str,
    start_time: float,
    end_time: float,
    output_file: str,
    quality: str = "720p",
    progress_placeholder = None,
) -> Optional[str]:
    """
    Professional 9:16 vertical rendering using FFmpeg.
    
    Quality Settings:
    - 720p: CRF 21, 192k audio (smaller file, faster rendering)
    - 1080p: CRF 18, 320k audio (better quality, larger file)
    
    Vertical mastering: scale to 1920px height, center-crop to 1080x1920
    """
    try:
        duration = float(end_time) - float(start_time)
        if duration <= 0:
            if progress_placeholder:
                progress_placeholder.error("❌ Invalid clip duration")
            return None
    except Exception:
        if progress_placeholder:
            progress_placeholder.error("❌ Invalid timestamps")
        return None
    
    # ROBUST FILE-SIZE VALIDATION: Check if input is 0 bytes or missing
    if not os.path.exists(input_file):
        if progress_placeholder:
            progress_placeholder.error("❌ Download produced no file. Streamlit Cloud server may be rate-limited by YouTube (HTTP 403 Forbidden).")
        return None
    
    file_size = os.path.getsize(input_file)
    if file_size == 0:
        if progress_placeholder:
            progress_placeholder.error("❌ Download was empty (0 bytes). YouTube is blocking this Streamlit Cloud IP. Please try again later or use a different video.")
        return None
    
    # Quality-based encoding parameters
    if quality == "1080p" or "1080" in str(quality):
        video_crf = "18"
        audio_bitrate = "320k"
    else:
        video_crf = "21"
        audio_bitrate = "192k"
    
    # Professional FFmpeg command for 9:16 vertical mastering
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_file,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", video_crf,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        output_file,
    ]
    
    if progress_placeholder:
        progress_placeholder.info(f"✨ Rendering {quality} vertical short...")
    
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError:
        if progress_placeholder:
            progress_placeholder.error("❌ ffmpeg not found. Install ffmpeg and retry.")
        return None
    except Exception as e:
        if progress_placeholder:
            progress_placeholder.error(f"❌ FFmpeg error: {str(e)}")
        return None
    
    if proc.returncode != 0:
        if progress_placeholder:
            progress_placeholder.error("❌ FFmpeg rendering failed")
        return None
    
    if os.path.exists(output_file):
        if progress_placeholder:
            progress_placeholder.success("✅ Vertical mastering complete")
        return output_file
    else:
        if progress_placeholder:
            progress_placeholder.error("❌ Output file not created")
        return None


def create_viral_short(
    video_url: str,
    start_time: float,
    end_time: float,
    quality: str = "720p",
    progress_placeholder = None,
    status_placeholder = None,
) -> Optional[Dict[str, str]]:
    """
    Clip directly from source using yt-dlp + ffmpeg downloader.
    """
    start_seconds = max(0.0, float(start_time))
    duration = max(0.0, float(end_time) - float(start_time))
    if duration <= 0:
        return None

    output_file = temp_path("short_clip.mp4")
    input_url = video_url if video_url.startswith("http") else f"https://www.youtube.com/watch?v={extract_video_id(video_url) or ''}"
    start_stamp = format_hhmmss(start_seconds)
    end_stamp = format_hhmmss(start_seconds + duration)

    try:
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception:
        pass

    cmd = [
        "yt-dlp",
        "--no-check-certificate",
        "--user-agent", DEFAULT_USER_AGENT,
        "--referer", "https://www.youtube.com/",
        "--downloader", "ffmpeg",
        "--downloader-args", f"ffmpeg:-ss {start_stamp} -to {end_stamp} -c copy",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", output_file,
        input_url,
    ]
    current_cookies_path = app_path("cookies.txt")
    if os.path.exists(current_cookies_path):
        cmd[1:1] = ["--cookies", current_cookies_path]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None

    if proc.returncode != 0:
        store_yt_dlp_error(proc.stdout, proc.stderr)
        return None

    if os.path.exists(output_file) and os.path.getsize(output_file) > 100000:
        return {"mode": "clip", "value": output_file}
    return None


def get_direct_stream_url(video_url: str) -> Tuple[bool, str]:
    """
    Extract direct stream URL from YouTube via yt-dlp (-g).
    """
    cmd = [
        "yt-dlp",
        "--no-check-certificate",
        "--user-agent", DEFAULT_USER_AGENT,
        "-g",
        video_url,
    ]
    cookies_path = app_path("cookies.txt")
    if os.path.exists(cookies_path):
        cmd[1:1] = ["--cookies", cookies_path]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        return False, f"تعذر تشغيل yt-dlp: {exc}"

    if proc.returncode != 0:
        store_yt_dlp_error(proc.stdout, proc.stderr)
        error_tail = tail_text(f"{proc.stdout}\n{proc.stderr}", max_lines=15)
        return False, f"فشل استخراج الرابط المباشر عبر yt-dlp: {error_tail or 'Unknown error'}"

    # yt-dlp قد يرجع عدة روابط (فيديو/صوت)، نأخذ أول رابط صالح.
    candidates = [line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("http")]
    if not candidates:
        return False, "لم يرجع yt-dlp رابطًا مباشرًا صالحًا."
    return True, candidates[0]


def read_essential_youtube_cookies() -> str:
    """
    Extract only essential YouTube cookies from Netscape cookies.txt.
    """
    cookies_path = app_path("cookies.txt")
    if not os.path.exists(cookies_path):
        return ""

    required_names = {
        "VISITOR_INFO1_LIVE",
        "HSID",
        "SSID",
        "APISID",
        "SAPISID",
        "LOGIN_INFO",
        "__Secure-3PSID",
        "__Secure-3PAPISID",
    }
    cookie_map: Dict[str, str] = {}
    try:
        with open(cookies_path, "r", encoding="utf-8", errors="ignore") as cookie_file:
            for raw_line in cookie_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                name = parts[5].strip()
                value = parts[6].strip()
                if name in required_names and value:
                    cookie_map[name] = value
    except Exception:
        return ""

    ordered_names = [
        "VISITOR_INFO1_LIVE",
        "HSID",
        "SSID",
        "APISID",
        "SAPISID",
        "LOGIN_INFO",
        "__Secure-3PSID",
        "__Secure-3PAPISID",
    ]
    return "; ".join([f"{name}={cookie_map[name]}" for name in ordered_names if name in cookie_map])


def stream_clip_with_ytdlp_to_memory(
    video_url: str,
    start_time: float,
    end_time: float,
) -> Tuple[bool, Optional[io.BytesIO], str]:
    """
    Primary strategy: yt-dlp download-sections to temp file, then BytesIO.
    """
    start_value = max(0.0, float(start_time))
    end_value = max(start_value + 1.0, float(end_time))
    section_spec = f"*{format_hhmmss(start_value)}-{format_hhmmss(end_value)}"
    temp_output_path = Path("/tmp/final_clip.mp4")
    if not temp_output_path.parent.exists():
        temp_output_path = Path(temp_path("final_clip.mp4"))
    output_file = str(temp_output_path)

    try:
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception:
        pass

    cmd = [
        "yt-dlp",
        "--no-check-certificate",
        "--quiet",
        "--no-warnings",
        "--no-progress",
        "--extractor-args", "youtube:player_client=android_vr,web_safari",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "--download-sections", section_spec,
        "--force-overwrites",
        "-f", "best[ext=mp4]/best",
        "-o", output_file,
        video_url,
    ]
    cookies_path = app_path("cookies.txt")
    if os.path.exists(cookies_path):
        cmd[1:1] = ["--cookies", cookies_path]

    # Re-read cookies path before every request to keep credentials fresh.
    try:
        proc = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        return False, None, f"فشل تشغيل yt-dlp: {exc}"

    if proc.returncode != 0:
        ytdlp_tail = tail_text((proc.stderr or b"").decode("utf-8", errors="ignore"), max_lines=20)
        return False, None, f"فشل yt-dlp أثناء قص المقطع: {ytdlp_tail or 'Unknown error'}"

    if not os.path.exists(output_file):
        return False, None, "yt-dlp انتهى لكن لم يُنشئ الملف المؤقت."
    if os.path.getsize(output_file) <= 0:
        return False, None, "الملف المؤقت الناتج فارغ."

    try:
        with open(output_file, "rb") as generated_clip:
            clip_bytes = generated_clip.read()
    except Exception as exc:
        return False, None, f"تعذر قراءة الملف المؤقت: {exc}"
    finally:
        try:
            os.remove(output_file)
        except Exception:
            pass

    clip_buffer = io.BytesIO(clip_bytes)
    clip_buffer.seek(0)
    return True, clip_buffer, ""


def cut_direct_stream_to_memory(
    direct_url: str,
    start_time: float,
    duration: float,
    source_video_url: Optional[str] = None,
) -> Tuple[bool, Optional[io.BytesIO], str]:
    """
    Clip direct stream URL in-memory using FFmpeg pipe output.
    """
    if duration <= 0:
        return False, None, "مدة القص غير صالحة."

    cleaned_cookies = read_essential_youtube_cookies()
    ffmpeg_headers = f"Cookie: {cleaned_cookies}\r\nUser-Agent: {DEFAULT_USER_AGENT}\r\n"

    cmd = [
        "ffmpeg",
        "-headers", ffmpeg_headers,
        "-ss", str(max(0.0, float(start_time))),
        "-i", direct_url,
        "-t", str(float(duration)),
        "-c", "copy",
        "-f", "mp4",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError:
        return False, None, "ffmpeg غير مثبت أو غير موجود في PATH."
    except Exception as exc:
        return False, None, f"فشل تنفيذ FFmpeg: {exc}"

    if proc.returncode != 0:
        ffmpeg_tail = tail_text(proc.stderr.decode("utf-8", errors="ignore"), max_lines=20)
        if source_video_url:
            fallback_ok, fallback_buffer, fallback_error = stream_clip_with_ytdlp_to_memory(
                video_url=source_video_url,
                start_time=max(0.0, float(start_time)),
                end_time=max(0.0, float(start_time)) + float(duration),
            )
            if fallback_ok and fallback_buffer is not None:
                fallback_buffer.seek(0)
                return True, fallback_buffer, ""
            return False, None, f"فشل FFmpeg ثم fallback yt-dlp: {fallback_error}"
        return False, None, f"فشل FFmpeg أثناء القص: {ffmpeg_tail or 'Unknown error'}"

    if not proc.stdout:
        return False, None, "FFmpeg انتهى دون إخراج بيانات MP4."

    clip_buffer = io.BytesIO(proc.stdout)
    clip_buffer.seek(0)
    return True, clip_buffer, ""


# ============= STAGE 1: INPUT & ANALYSIS =============
def render_stage_1():
    """STAGE 1: Professional input form with video analysis."""
    st.markdown('<div class="stage-badge">📊 STAGE 1: INPUT & ANALYSIS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Settings")
        st.session_state.quality = st.selectbox(
            "Video Quality",
            ["720p", "1080p (HD)"],
            index=0,
        )
        
        st.session_state.use_custom_keywords = st.checkbox(
            "Use custom keywords",
            value=False,
        )
        if st.session_state.use_custom_keywords:
            st.session_state.custom_keywords = st.text_area(
                "Keywords (comma-separated)",
                placeholder="viral, trending, epic",
                height=80,
            )
    
    with col2:
        st.markdown("### 🎥 YouTube Video Analyzer")
        st.markdown("Paste a YouTube URL to extract viral moments using AI Intelligence.")
        
        st.session_state.url = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://youtube.com/watch?v=... or dQw4w9WgXcQ",
        )
        
        col_lang, col_ratio = st.columns(2)
        with col_lang:
            lang_input = st.multiselect(
                "Transcript Language",
                ["English", "Spanish", "French", "German", "Portuguese"],
                default=["English"],
            )
            lang_map = {
                "English": "en", "Spanish": "es", "French": "fr",
                "German": "de", "Portuguese": "pt",
            }
            st.session_state.languages = [lang_map[l] for l in lang_input]
        
        with col_ratio:
            st.markdown("**Format: 9:16 (Vertical)**")
            st.info("Vertical shorts are optimized for mobile viewing")
        
        if st.button("🔍 Analyze Video", use_container_width=True):
            if not st.session_state.url:
                st.error("❌ Enter a YouTube URL or video ID")
            else:
                video_id = extract_video_id(st.session_state.url)
                if not video_id:
                    st.error("❌ Invalid YouTube URL")
                else:
                    video_url = st.session_state.url if st.session_state.url.startswith("http") else f"https://www.youtube.com/watch?v={video_id}"
                    
                    # Create progress indicators
                    status_placeholder = st.status("🔄 Fetching transcript...", expanded=True)
                    
                    try:
                        with status_placeholder:
                            # Try fast transcript fetch with 10-second timeout
                            import threading
                            transcript = [None]
                            transcript_source = ["none"]
                            
                            def fetch_transcript_thread():
                                try:
                                    t, s = _get_transcript_smart_result(video_id)
                                    transcript[0] = t
                                    transcript_source[0] = s
                                except Exception:
                                    transcript[0] = None
                                    transcript_source[0] = "none"
                            
                            thread = threading.Thread(target=fetch_transcript_thread, daemon=True)
                            thread.start()
                            thread.join(timeout=10)  # 10-second timeout
                            
                            # Check if transcript was fetched
                            if transcript[0]:
                                st.write(f"✅ Successfully fetched {len(transcript[0])} transcript segments!")
                                st.session_state.video_id = video_id
                                st.session_state.video_url = video_url
                                st.session_state.transcript = transcript[0]
                                st.session_state.transcript_source = transcript_source[0]
                                st.session_state.transcript_text = format_transcript(transcript[0], show_timestamps=True)
                                st.session_state.viral_moments = analyze_with_ai(
                                    transcript[0],
                                    custom_keywords=parse_custom_keywords(st.session_state.custom_keywords) if st.session_state.use_custom_keywords else None,
                                    top_n=3,
                                )
                                st.session_state.fast_mode_enabled = False
                                status_placeholder.update(label="✅ Transcript loaded!", state="complete")
                            else:
                                # FAST ENGINE MODE: Use metadata instead of transcript
                                st.write("⏳ YouTube is slow... switching to Fast Engine!")
                                status_placeholder.update(label="🚀 Using Fast Engine Mode", state="running")
                                
                                metadata = get_fast_engine_metadata(video_id)
                                if metadata:
                                    st.write(f"📹 Video: {metadata['title']}")
                                    st.write(f"⏱️ Duration: {metadata['duration']//60} minutes")
                                    
                                    # Generate quick viral moments from description
                                    quick_moments = generate_viral_moments_from_description(
                                        metadata["description"],
                                        metadata["title"]
                                    )
                                    
                                    st.session_state.video_id = video_id
                                    st.session_state.video_url = video_url
                                    st.session_state.transcript = None  # No full transcript in fast mode
                                    st.session_state.transcript_text = f"**المحرك السريع (Fast Engine)**\n\n{metadata['title']}\n\n{metadata['description']}"
                                    st.session_state.viral_moments = quick_moments
                                    st.session_state.fast_mode_enabled = True
                                    st.write("✅ Ready to download!")
                                    status_placeholder.update(label="✅ Fast Engine ready!", state="complete")
                                else:
                                    status_placeholder.update(label="⚠️ No transcript available", state="error")
                                    st.warning("Try another video")
                                    return
                            
                            st.session_state.stage = 2
                            st.rerun()
                    except Exception as e:
                        status_placeholder.update(label="❌ Error", state="error")
                        st.error(f"❌ Unexpected error: {str(e)}")
                        st.write("Please try again or use a different video.")


# ============= STAGE 2: SELECTION & PREVIEW =============
def render_stage_2():
    """STAGE 2: Two-column layout with transcript and viral moments."""
    st.markdown('<div class="stage-badge">🤖 STAGE 2: SELECTION & PREVIEW</div>', unsafe_allow_html=True)
    
    # Show Fast Mode indicator if enabled
    if st.session_state.fast_mode_enabled:
        st.info("⚡ **Fast Engine Mode** - جاري استخدام المحرك السريع للتحليل السريع")
    
    # Navigation
    col_nav1, col_nav2 = st.columns([1, 10])
    with col_nav1:
        if st.button("← Back"):
            st.session_state.stage = 1
            st.rerun()
    with col_nav2:
        if st.button("🔄 Reset"):
            for key in list(st.session_state.keys()):
                st.session_state[key] = None
            st.session_state.stage = 1
            st.rerun()
    
    st.markdown("---")
    
    # TWO-COLUMN LAYOUT: Transcript (left) | Viral Scores (right)
    col_transcript, col_moments = st.columns([1.2, 1], gap="large")
    
    # ===== LEFT COLUMN: Transcript =====
    with col_transcript:
        st.markdown("### 📜 Generated Transcript")
        st.markdown('<div class="transcript-box">' + st.session_state.transcript_text.replace('\n', '<br>') + '</div>', 
                   unsafe_allow_html=True)
    
    # ===== RIGHT COLUMN: Viral Moments =====
    with col_moments:
        st.markdown("### 🔥 Viral Moments")
        st.markdown("*AI-detected high-potential clips*")
        
        if not st.session_state.viral_moments:
            with st.spinner("🔍 Scanning for viral moments..."):
                if st.session_state.transcript:
                    custom_kw = parse_custom_keywords(st.session_state.custom_keywords) if st.session_state.use_custom_keywords else None
                    st.session_state.viral_moments = analyze_with_ai(
                        st.session_state.transcript,
                        custom_keywords=custom_kw,
                        top_n=3
                    )
                else:
                    st.session_state.viral_moments = []
        
        if not st.session_state.viral_moments:
            st.info("💡 No viral moments detected. Try another video!")
        else:
            for idx, moment in enumerate(st.session_state.viral_moments, 1):
                with st.container(border=True):
                    st.markdown(f"#### #{idx} {moment['title']}")
                    st.caption(f"⏱️ {moment['timestamp']} | 45 seconds")
                    
                    # Viral Score
                    st.markdown(f'<div class="metric-value">{int(moment["viral_score"])}</div>', unsafe_allow_html=True)
                    st.progress(moment["viral_score"] / 100.0)
                    
                    st.markdown(f"*{moment['snippet']}*")
                    
                    if st.button(
                        f"✂️ Create Short #{idx}",
                        key=f"create_short_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_moment = moment
                        st.session_state.stage = 3
                        st.session_state.download_triggered = False
                        st.rerun()


# ============= STAGE 3: DIRECT PIPE DOWNLOAD =============
def render_stage_3():
    """STAGE 3: Server-side instant clipping and direct download button."""
    st.markdown("### 🚀 المرحلة النهائية: قص فوري + تحميل مباشر")

    video_id = st.session_state.video_id
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    selected = st.session_state.selected_moment

    st.markdown(f"**اللقطة المختارة:** {selected['title']} | {selected['timestamp']} | درجة الفيروسية: {int(selected['viral_score'])}/100")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← العودة للقائمة", use_container_width=True):
            st.session_state.stage = 2
            st.rerun()
    with col3:
        if st.button("🔄 فيديو جديد", use_container_width=True):
            st.session_state.stage = 1
            for key in ["video_id", "url", "transcript", "viral_moments", "selected_moment", "output_video", "download_link"]:
                st.session_state[key] = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 📥 التحميل الذكي من المتصفح")
    quality_choice = st.radio(
        "اختر جودة الفيديو",
        ["720p (موصى به)", "1080p (HD)", "480p (سريع)"],
        index=0,
        horizontal=True,
    )
    quality_map = {
        "720p (موصى به)": "720p",
        "1080p (HD)": "1080p",
        "480p (سريع)": "480p",
    }
    selected_quality = quality_map[quality_choice]

    start_time = float(selected.get("start_time", 0.0))
    end_time = float(selected.get("end_time", start_time + 45.0))
    clip_duration = max(1.0, end_time - start_time)
    start_seconds = max(0, int(start_time))
    end_seconds = max(start_seconds + 1, int(end_time))

    st.info("⚡ السيرفر سيقص المقطع المطلوب مباشرة من يوتيوب ويرسله فورًا كملف MP4.")
    st.markdown("### 🎬 معاينة داخل الصفحة")
    st.video(f"{video_url}&t={start_seconds}s")

    if "stage3_clip_bytes" not in st.session_state:
        st.session_state.stage3_clip_bytes = None
    if "stage3_clip_filename" not in st.session_state:
        st.session_state.stage3_clip_filename = None
    if "stage3_clip_key" not in st.session_state:
        st.session_state.stage3_clip_key = None

    clip_key = f"{video_id}:{start_seconds}:{end_seconds}:{selected_quality}"
    if st.session_state.stage3_clip_key != clip_key:
        st.session_state.stage3_clip_key = clip_key
        st.session_state.stage3_clip_bytes = None
        st.session_state.stage3_clip_filename = None

    st.markdown("### 📥 تحميل MP4 (yt-dlp Piping)")
    if st.button("⚡ جهّز القص المباشر للتحميل", use_container_width=True):
        with st.spinner("جاري تحميل وقص المقطع عبر yt-dlp مباشرة إلى الذاكرة..."):
            ok_clip, clip_buffer, clip_error = stream_clip_with_ytdlp_to_memory(
                video_url=video_url,
                start_time=start_time,
                end_time=end_time,
            )
            if not ok_clip or clip_buffer is None:
                st.error(clip_error or "فشل تجهيز المقطع.")
            else:
                st.session_state.stage3_clip_bytes = clip_buffer.getvalue()
                st.session_state.stage3_clip_filename = f"viral_clip_{video_id}_{start_seconds}_{end_seconds}.mp4"
                st.success("✅ المقطع جاهز. اضغط زر التحميل الآن.")

    if st.session_state.stage3_clip_bytes:
        st.markdown("### ▶️ معاينة المقطع الناتج")
        st.video(st.session_state.stage3_clip_bytes)
        st.download_button(
            label="⬇️ تنزيل المقطع الآن",
            data=io.BytesIO(st.session_state.stage3_clip_bytes),
            file_name=st.session_state.stage3_clip_filename or "viral_clip.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    st.caption(f"توقيت اللقطة المقترح: من {start_seconds}s إلى {end_seconds}s | الجودة المختارة: {selected_quality} | مدة القص: {int(clip_duration)}s")


# ============= MAIN APP FLOW =============
if st.session_state.stage == 1:
    render_stage_1()
elif st.session_state.stage == 2:
    render_stage_2()
elif st.session_state.stage == 3:
    render_stage_3()

st.markdown("---")

# ============= PROFESSIONAL FOOTER =============
st.markdown(
    '<div class="footer-branding">Designed & Developed by MR. HAMZA AIT TALEB | '
    'Powered by GitHub Education</div>',
    unsafe_allow_html=True
)
