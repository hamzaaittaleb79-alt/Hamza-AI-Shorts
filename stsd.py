"""
================================================================================
🎬 ViraFlow - Professional Short-Form Engine (SaaS Edition)
================================================================================
Production-ready Streamlit application for AI-driven YouTube short creation.
Optimized for Streamlit Cloud deployment with cloud-safe paths and headless mode.

Author: MR. HAMZA AIT TALEB
GitHub Education Partner | AI Intelligence Engine
================================================================================
"""
import random
import requests
import re
import json
import os
import shutil
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import yt_dlp

# ============= OPTIONAL IMPORTS (Graceful Fallback) =============
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    WhisperModel = None

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


# ============= CLOUD-SAFE PATH CONFIGURATION =============
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
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
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "-o", output_template,
        # استخدام العميل المكتبي العادي عبر الوسيط
        "--extractor-args", "youtube:player_client=web",
        "--force-ipv4",
    ]
    
    # إضافة ملف الكوكيز إذا كان موجوداً (اختياري مع الوسيط)
    if os.path.exists("cookies.txt"):
        cmd.extend(["--cookies", "cookies.txt"])
    
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


# ============= TRANSCRIPTION INTELLIGENCE ENGINE =============
@st.cache_resource
def load_whisper_model():
    """
    Load faster-whisper model once per session.
    Uses base model with int8 quantization for memory efficiency on Streamlit Cloud.
    """
    if not HAS_WHISPER:
        return None
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return None


def download_audio_for_transcription(video_url: str, video_id: str) -> Optional[str]:
    """Download audio from YouTube for Whisper transcription."""
    output_template = temp_path(f"whisper_audio_{video_id}.%(ext)s")
    cmd = build_yt_dlp_command(
        video_url,
        output_template,
        format_selector="bestaudio/best",
        extra_flags=["--extract-audio", "--audio-format", "mp3"],
    )
    
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    except Exception:
        return None
    
    if proc.returncode != 0:
        return None
    
    # Check for audio file in multiple formats
    for ext in ("mp3", "m4a", "webm", "mkv"):
        candidate = temp_path(f"whisper_audio_{video_id}.{ext}")
        if os.path.exists(candidate):
            return candidate
    
    return None


def transcribe_with_whisper(video_url: str, video_id: str) -> Optional[List[Dict]]:
    """
    Transcribe video using faster-whisper (local CPU) or OpenAI Whisper API.
    Gracefully falls back if services are unavailable.
    """
    # Try OpenAI API first if key is available
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        audio_path = download_audio_for_transcription(video_url, video_id)
        if audio_path:
            try:
                client = OpenAI()
                with open(audio_path, "rb") as audio_file:
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )
                
                segments = []
                for item in getattr(result, "segments", []) or []:
                    segments.append({
                        "text": item.get("text", "").strip(),
                        "start": float(item.get("start", 0.0)),
                        "duration": float(item.get("end", 0.0)) - float(item.get("start", 0.0)),
                    })
                return segments or None
            except Exception:
                pass
            finally:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
    
    # Fall back to local faster-whisper
    model = load_whisper_model()
    if not model:
        return None
    
    audio_path = download_audio_for_transcription(video_url, video_id)
    if not audio_path:
        return None
    
    try:
        segments, _info = model.transcribe(audio_path, vad_filter=True)
        transcript = []
        for seg in segments:
            transcript.append({
                "text": seg.text.strip(),
                "start": float(seg.start),
                "duration": float(seg.end - seg.start),
            })
        return transcript or None
    except Exception:
        return None
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass


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
        "custom_keywords": "",
        "use_custom_keywords": False,
        "transcription_status": None,
        "last_yt_dlp_error": "",
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
    Fetch transcript from YouTube captions API only.

    Uses cookies.txt when available to reduce 403/verification issues on
    Streamlit Cloud. Whisper fallback is handled by the caller.
    """
    try:
        cookies_path = app_path("cookies.txt")
        language_codes = tuple(languages) if languages else ("en",)
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

        transcript = api.fetch(video_id, languages=language_codes)
        return transcript.to_raw_data()
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception):
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


def format_hhmmss(seconds_value: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(float(seconds_value))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_custom_keywords(keywords_str: str) -> List[str]:
    """Parse comma-separated keywords into a list."""
    if not keywords_str or not keywords_str.strip():
        return []
    return [kw.strip().lower() for kw in keywords_str.split(",") if kw.strip()]


# ============= VERTICAL VIDEO RENDERING (9:16 Mastering) =============
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
) -> Optional[str]:
    """
    Production-grade short creation pipeline.
    
    Flow:
    1. Request clipped segment from Cobalt API (cloud-safe)
    2. Fallback to lower quality if API blocks/fails
    3. Return final output path ready for upload
    """
    output_file = app_path("final_viral_clip.mp4")
    temp_clip_path = temp_path("temp_clip.mp4")
    
    # Validate timestamps
    start_seconds = max(0.0, float(start_time))
    duration = max(0.0, float(end_time) - float(start_time))
    if duration <= 0:
        if status_placeholder:
            status_placeholder.error("❌ Invalid duration")
        return None

    def extract_stream_url(api_data: Any) -> Optional[str]:
        """Extract a direct stream/download URL from variable Cobalt API responses."""
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

    def run_cobalt_download_attempt(quality_level: int, attempt_label: str) -> Tuple[bool, str]:
        """Request clipped segment from Cobalt API and save it to temp_clip.mp4."""
        if os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except Exception:
                pass

        if status_placeholder:
            status_placeholder.info(f"🛡️ Downloading clipped segment via Cobalt ({attempt_label})...")

        payload = {
            "url": video_url,
            "sectionStart": round(start_seconds, 3),
            "sectionEnd": round(start_seconds + duration, 3),
            "quality": quality_level,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        }

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

    # Step 1: Cobalt API clipping with quality fallback
    last_error = ""
    for quality_level, attempt_label in [
        (720, "quality 720"),
        (480, "quality 480 fallback"),
        (360, "quality 360 emergency fallback"),
    ]:
        success, error_tail = run_cobalt_download_attempt(quality_level, attempt_label)
        if success:
            last_error = ""
            break
        last_error = error_tail
        st.session_state.last_yt_dlp_error = last_error

    if not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) == 0:
        if status_placeholder:
            if last_error and "403" in last_error:
                status_placeholder.error("❌ HTTP 403 Forbidden from upstream provider. Tried Cobalt fallback qualities but clip still failed.")
            else:
                status_placeholder.error("❌ Cobalt API clipping failed on all quality attempts.")
        if last_error:
            st.code(last_error, language="text")
        return None

    # Step 2: Save final clip directly (no heavy server-side reprocessing)
    if status_placeholder:
        status_placeholder.info("✅ Clipped segment downloaded from Cobalt API")

    try:
        shutil.copyfile(temp_clip_path, output_file)
    except Exception as e:
        if status_placeholder:
            status_placeholder.error(f"❌ Failed to finalize clip file: {e}")
        return None

    if progress_placeholder:
        progress_placeholder.success("✅ Clip ready for upload")

    return output_file


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
                    
                    with st.spinner("🔄 Analyzing Content..."):
                        transcript = fetch_transcript(video_id, languages=st.session_state.languages)
                        if transcript is None:
                            transcript = transcribe_with_whisper(video_url, video_id)
                        if transcript:
                            st.session_state.video_id = video_id
                            st.session_state.video_url = video_url
                            st.session_state.transcript = transcript
                            st.session_state.transcript_text = format_transcript(transcript, show_timestamps=True)
                            st.session_state.stage = 2
                            st.rerun()
                        else:
                            st.error("❌ Could not fetch transcript for this video")


# ============= STAGE 2: SELECTION & PREVIEW =============
def render_stage_2():
    """STAGE 2: Two-column layout with transcript and viral moments."""
    st.markdown('<div class="stage-badge">🤖 STAGE 2: SELECTION & PREVIEW</div>', unsafe_allow_html=True)
    
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
                custom_kw = parse_custom_keywords(st.session_state.custom_keywords) if st.session_state.use_custom_keywords else None
                st.session_state.viral_moments = find_viral_moments(
                    st.session_state.transcript,
                    custom_keywords=custom_kw,
                    top_n=3
                )
        
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
                        st.rerun()


# ============= STAGE 3: RENDERING =============
def render_stage_3():
    """واجهة مستر حمزة النهائية: تحميل مباشر بجودة عالية وتخطي حظر السيرفر"""
    st.markdown("### 🎯 المرحلة النهائية: تحميل الفيديو بالجودة المطلوبة")
    
    video_id = st.session_state.video_id
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    st.success("✅ تم تحليل الفيديو بنجاح! اختر الجودة والمحرك للتحميل المباشر:")

    # تصميم بطاقات التحميل
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### 🚀 المحرك السريع (Cobalt)")
            st.write("يدعم 1080p و 4K مباشرة")
            # هذا الرابط يفتح موقع تحميل احترافي ومجاني معبأ برابط الفيديو الخاص بك
            st.markdown(f'''
                <a href="https://cobalt.tools/" target="_blank">
                    <button style="width:100%; background-color:#FFD700; border:none; color:black; padding:12px; cursor:pointer; border-radius:8px; font-weight:bold;">
                        فتح محرك Cobalt للتحميل
                    </button>
                </a>
            ''', unsafe_allow_html=True)
            st.caption("انسخ الرابط وضعه في Cobalt للحصول على أعلى جودة.")

    with col2:
        with st.container(border=True):
            st.markdown("#### ⚡ المحرك الاحترافي (SaveFrom)")
            st.write("تحميل مباشر وسهل")
            st.markdown(f'''
                <a href="https://en.savefrom.net/18/#url={video_url}" target="_blank">
                    <button style="width:100%; background-color:#00E676; border:none; color:white; padding:12px; cursor:pointer; border-radius:8px; font-weight:bold;">
                        تحميل عبر SaveFrom
                    </button>
                </a>
            ''', unsafe_allow_html=True)
            st.caption("سيفتح الموقع والرابط جاهز للتحميل فوراً.")

    st.info(f"🔗 رابط الفيديو الخاص بك: `{video_url}`")
    
    if st.button("🔄 تحليل فيديو آخر"):
        st.session_state.stage = 1
        st.session_state.video_id = None
        st.rerun()
    
    st.markdown("---")
    
    selected = st.session_state.selected_moment
    st.markdown(f"### ✂️ Creating: **{selected['title']}**")
    st.markdown(f"*Timestamp: {selected['timestamp']} | Viral Score: {int(selected['viral_score'])}/100 | Quality: {st.session_state.quality}*")
    
    st.markdown("---")
    
    # Check if already rendered
    output_path = app_path("final_viral_clip.mp4")
    if os.path.exists(output_path):
        st.success("✨ SUCCESS! Your viral short is ready!")
        st.video(output_path)
        
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Video",
                data=f.read(),
                file_name="final_viral_clip.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎬 Create Another", use_container_width=True):
                st.session_state.stage = 1
                st.rerun()
        with col2:
            if st.button("🎯 View More Clips", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()
        return
    
    # Rendering placeholders
    status_container = st.status("🛡️ Initializing rendering...", expanded=True)
    progress_slot = st.empty()
    result_container = st.empty()
    
    video_url = st.session_state.get("video_url") or f"https://www.youtube.com/watch?v={st.session_state.video_id}"
    
    try:
        with status_container:
            result = create_viral_short(
                video_url,
                selected["start_time"],
                selected["end_time"],
                quality=st.session_state.quality,
                progress_placeholder=progress_slot,
                status_placeholder=status_container,
            )
        
        if result and os.path.exists(result):
            status_container.update(label="✅ Rendering complete!", state="complete", expanded=False)
            time.sleep(1)
            
            with result_container.container():
                st.success("✨ SUCCESS! Your viral short is ready!")
                st.video(result)
                
                with open(result, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=f.read(),
                        file_name="final_viral_clip.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎬 Create Another", use_container_width=True):
                        st.session_state.stage = 1
                        st.rerun()
                with col2:
                    if st.button("🎯 View More Clips", use_container_width=True):
                        st.session_state.stage = 2
                        st.rerun()
        else:
            status_container.update(label="❌ Rendering failed", state="error", expanded=True)
            result_container.error("❌ Failed to create short. Download may be blocked on Streamlit Cloud (HTTP 403 Forbidden).")
            last_error = st.session_state.get("last_yt_dlp_error", "")
            if last_error:
                st.code(last_error, language="text")
    
    except Exception as e:
        status_container.update(label="❌ Error", state="error", expanded=True)
        result_container.error(f"❌ Unexpected error: {str(e)}")


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
)"""
================================================================================
🎬 ViraFlow - Professional Short-Form Engine (SaaS Edition)
================================================================================
Production-ready Streamlit application for AI-driven YouTube short creation.
Optimized for Streamlit Cloud deployment with cloud-safe paths and headless mode.

Author: MR. HAMZA AIT TALEB
GitHub Education Partner | AI Intelligence Engine
================================================================================
"""
import random
import requests
import re
import json
import os
import shutil
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import yt_dlp

# ============= OPTIONAL IMPORTS (Graceful Fallback) =============
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    WhisperModel = None

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


# ============= CLOUD-SAFE PATH CONFIGURATION =============
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
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
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "-o", output_template,
        # استخدام العميل المكتبي العادي عبر الوسيط
        "--extractor-args", "youtube:player_client=web",
        "--force-ipv4",
    ]
    
    # إضافة ملف الكوكيز إذا كان موجوداً (اختياري مع الوسيط)
    if os.path.exists("cookies.txt"):
        cmd.extend(["--cookies", "cookies.txt"])
    
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


# ============= TRANSCRIPTION INTELLIGENCE ENGINE =============
@st.cache_resource
def load_whisper_model():
    """
    Load faster-whisper model once per session.
    Uses base model with int8 quantization for memory efficiency on Streamlit Cloud.
    """
    if not HAS_WHISPER:
        return None
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return None


def download_audio_for_transcription(video_url: str, video_id: str) -> Optional[str]:
    """Download audio from YouTube for Whisper transcription."""
    output_template = temp_path(f"whisper_audio_{video_id}.%(ext)s")
    cmd = build_yt_dlp_command(
        video_url,
        output_template,
        format_selector="bestaudio/best",
        extra_flags=["--extract-audio", "--audio-format", "mp3"],
    )
    
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    except Exception:
        return None
    
    if proc.returncode != 0:
        return None
    
    # Check for audio file in multiple formats
    for ext in ("mp3", "m4a", "webm", "mkv"):
        candidate = temp_path(f"whisper_audio_{video_id}.{ext}")
        if os.path.exists(candidate):
            return candidate
    
    return None


def transcribe_with_whisper(video_url: str, video_id: str) -> Optional[List[Dict]]:
    """
    Transcribe video using faster-whisper (local CPU) or OpenAI Whisper API.
    Gracefully falls back if services are unavailable.
    """
    # Try OpenAI API first if key is available
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        audio_path = download_audio_for_transcription(video_url, video_id)
        if audio_path:
            try:
                client = OpenAI()
                with open(audio_path, "rb") as audio_file:
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )
                
                segments = []
                for item in getattr(result, "segments", []) or []:
                    segments.append({
                        "text": item.get("text", "").strip(),
                        "start": float(item.get("start", 0.0)),
                        "duration": float(item.get("end", 0.0)) - float(item.get("start", 0.0)),
                    })
                return segments or None
            except Exception:
                pass
            finally:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
    
    # Fall back to local faster-whisper
    model = load_whisper_model()
    if not model:
        return None
    
    audio_path = download_audio_for_transcription(video_url, video_id)
    if not audio_path:
        return None
    
    try:
        segments, _info = model.transcribe(audio_path, vad_filter=True)
        transcript = []
        for seg in segments:
            transcript.append({
                "text": seg.text.strip(),
                "start": float(seg.start),
                "duration": float(seg.end - seg.start),
            })
        return transcript or None
    except Exception:
        return None
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass


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
        "custom_keywords": "",
        "use_custom_keywords": False,
        "transcription_status": None,
        "last_yt_dlp_error": "",
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
    Fetch transcript from YouTube captions API only.

    Uses cookies.txt when available to reduce 403/verification issues on
    Streamlit Cloud. Whisper fallback is handled by the caller.
    """
    try:
        cookies_path = app_path("cookies.txt")
        language_codes = tuple(languages) if languages else ("en",)
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

        transcript = api.fetch(video_id, languages=language_codes)
        return transcript.to_raw_data()
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception):
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


def format_hhmmss(seconds_value: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(float(seconds_value))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_custom_keywords(keywords_str: str) -> List[str]:
    """Parse comma-separated keywords into a list."""
    if not keywords_str or not keywords_str.strip():
        return []
    return [kw.strip().lower() for kw in keywords_str.split(",") if kw.strip()]


# ============= VERTICAL VIDEO RENDERING (9:16 Mastering) =============
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
) -> Optional[str]:
    """
    Production-grade short creation pipeline.
    
    Flow:
    1. Request clipped segment from Cobalt API (cloud-safe)
    2. Fallback to lower quality if API blocks/fails
    3. Return final output path ready for upload
    """
    output_file = app_path("final_viral_clip.mp4")
    temp_clip_path = temp_path("temp_clip.mp4")
    
    # Validate timestamps
    start_seconds = max(0.0, float(start_time))
    duration = max(0.0, float(end_time) - float(start_time))
    if duration <= 0:
        if status_placeholder:
            status_placeholder.error("❌ Invalid duration")
        return None

    def extract_stream_url(api_data: Any) -> Optional[str]:
        """Extract a direct stream/download URL from variable Cobalt API responses."""
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

    def run_cobalt_download_attempt(quality_level: int, attempt_label: str) -> Tuple[bool, str]:
        """Request clipped segment from Cobalt API and save it to temp_clip.mp4."""
        if os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except Exception:
                pass

        if status_placeholder:
            status_placeholder.info(f"🛡️ Downloading clipped segment via Cobalt ({attempt_label})...")

        payload = {
            "url": video_url,
            "sectionStart": round(start_seconds, 3),
            "sectionEnd": round(start_seconds + duration, 3),
            "quality": quality_level,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        }

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

    # Step 1: Cobalt API clipping with quality fallback
    last_error = ""
    for quality_level, attempt_label in [
        (720, "quality 720"),
        (480, "quality 480 fallback"),
        (360, "quality 360 emergency fallback"),
    ]:
        success, error_tail = run_cobalt_download_attempt(quality_level, attempt_label)
        if success:
            last_error = ""
            break
        last_error = error_tail
        st.session_state.last_yt_dlp_error = last_error

    if not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) == 0:
        if status_placeholder:
            if last_error and "403" in last_error:
                status_placeholder.error("❌ HTTP 403 Forbidden from upstream provider. Tried Cobalt fallback qualities but clip still failed.")
            else:
                status_placeholder.error("❌ Cobalt API clipping failed on all quality attempts.")
        if last_error:
            st.code(last_error, language="text")
        return None

    # Step 2: Save final clip directly (no heavy server-side reprocessing)
    if status_placeholder:
        status_placeholder.info("✅ Clipped segment downloaded from Cobalt API")

    try:
        shutil.copyfile(temp_clip_path, output_file)
    except Exception as e:
        if status_placeholder:
            status_placeholder.error(f"❌ Failed to finalize clip file: {e}")
        return None

    if progress_placeholder:
        progress_placeholder.success("✅ Clip ready for upload")

    return output_file


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
                    
                    with st.spinner("🔄 Analyzing Content..."):
                        transcript = fetch_transcript(video_id, languages=st.session_state.languages)
                        if transcript is None:
                            transcript = transcribe_with_whisper(video_url, video_id)
                        if transcript:
                            st.session_state.video_id = video_id
                            st.session_state.video_url = video_url
                            st.session_state.transcript = transcript
                            st.session_state.transcript_text = format_transcript(transcript, show_timestamps=True)
                            st.session_state.stage = 2
                            st.rerun()
                        else:
                            st.error("❌ Could not fetch transcript for this video")


# ============= STAGE 2: SELECTION & PREVIEW =============
def render_stage_2():
    """STAGE 2: Two-column layout with transcript and viral moments."""
    st.markdown('<div class="stage-badge">🤖 STAGE 2: SELECTION & PREVIEW</div>', unsafe_allow_html=True)
    
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
                custom_kw = parse_custom_keywords(st.session_state.custom_keywords) if st.session_state.use_custom_keywords else None
                st.session_state.viral_moments = find_viral_moments(
                    st.session_state.transcript,
                    custom_keywords=custom_kw,
                    top_n=3
                )
        
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
                        st.rerun()


# ============= STAGE 3: RENDERING =============
def render_stage_3():
    """واجهة مستر حمزة النهائية: تحميل مباشر بجودة عالية وتخطي حظر السيرفر"""
    st.markdown("### 🎯 المرحلة النهائية: تحميل الفيديو بالجودة المطلوبة")
    
    video_id = st.session_state.video_id
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    st.success("✅ تم تحليل الفيديو بنجاح! اختر الجودة والمحرك للتحميل المباشر:")

    # تصميم بطاقات التحميل
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### 🚀 المحرك السريع (Cobalt)")
            st.write("يدعم 1080p و 4K مباشرة")
            # هذا الرابط يفتح موقع تحميل احترافي ومجاني معبأ برابط الفيديو الخاص بك
            st.markdown(f'''
                <a href="https://cobalt.tools/" target="_blank">
                    <button style="width:100%; background-color:#FFD700; border:none; color:black; padding:12px; cursor:pointer; border-radius:8px; font-weight:bold;">
                        فتح محرك Cobalt للتحميل
                    </button>
                </a>
            ''', unsafe_allow_html=True)
            st.caption("انسخ الرابط وضعه في Cobalt للحصول على أعلى جودة.")

    with col2:
        with st.container(border=True):
            st.markdown("#### ⚡ المحرك الاحترافي (SaveFrom)")
            st.write("تحميل مباشر وسهل")
            st.markdown(f'''
                <a href="https://en.savefrom.net/18/#url={video_url}" target="_blank">
                    <button style="width:100%; background-color:#00E676; border:none; color:white; padding:12px; cursor:pointer; border-radius:8px; font-weight:bold;">
                        تحميل عبر SaveFrom
                    </button>
                </a>
            ''', unsafe_allow_html=True)
            st.caption("سيفتح الموقع والرابط جاهز للتحميل فوراً.")

    st.info(f"🔗 رابط الفيديو الخاص بك: `{video_url}`")
    
    if st.button("🔄 تحليل فيديو آخر"):
        st.session_state.stage = 1
        st.session_state.video_id = None
        st.rerun()
    
    st.markdown("---")
    
    selected = st.session_state.selected_moment
    st.markdown(f"### ✂️ Creating: **{selected['title']}**")
    st.markdown(f"*Timestamp: {selected['timestamp']} | Viral Score: {int(selected['viral_score'])}/100 | Quality: {st.session_state.quality}*")
    
    st.markdown("---")
    
    # Check if already rendered
    output_path = app_path("final_viral_clip.mp4")
    if os.path.exists(output_path):
        st.success("✨ SUCCESS! Your viral short is ready!")
        st.video(output_path)
        
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Video",
                data=f.read(),
                file_name="final_viral_clip.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎬 Create Another", use_container_width=True):
                st.session_state.stage = 1
                st.rerun()
        with col2:
            if st.button("🎯 View More Clips", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()
        return
    
    # Rendering placeholders
    status_container = st.status("🛡️ Initializing rendering...", expanded=True)
    progress_slot = st.empty()
    result_container = st.empty()
    
    video_url = st.session_state.get("video_url") or f"https://www.youtube.com/watch?v={st.session_state.video_id}"
    
    try:
        with status_container:
            result = create_viral_short(
                video_url,
                selected["start_time"],
                selected["end_time"],
                quality=st.session_state.quality,
                progress_placeholder=progress_slot,
                status_placeholder=status_container,
            )
        
        if result and os.path.exists(result):
            status_container.update(label="✅ Rendering complete!", state="complete", expanded=False)
            time.sleep(1)
            
            with result_container.container():
                st.success("✨ SUCCESS! Your viral short is ready!")
                st.video(result)
                
                with open(result, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=f.read(),
                        file_name="final_viral_clip.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎬 Create Another", use_container_width=True):
                        st.session_state.stage = 1
                        st.rerun()
                with col2:
                    if st.button("🎯 View More Clips", use_container_width=True):
                        st.session_state.stage = 2
                        st.rerun()
        else:
            status_container.update(label="❌ Rendering failed", state="error", expanded=True)
            result_container.error("❌ Failed to create short. Download may be blocked on Streamlit Cloud (HTTP 403 Forbidden).")
            last_error = st.session_state.get("last_yt_dlp_error", "")
            if last_error:
                st.code(last_error, language="text")
    
    except Exception as e:
        status_container.update(label="❌ Error", state="error", expanded=True)
        result_container.error(f"❌ Unexpected error: {str(e)}")


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
