"""
🚀 Darshj.AI - Ultimate Transcription Dashboard
================================================
Transcribe like a PRO with AI-powered magic! ✨

Standalone application — no Jupyter required.
"""

import os
import sys
import json
import time
import tempfile
import platform
import random
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

import gradio as gr
import whisper
import ffmpeg
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import humanize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
WHISPER_MODEL = os.environ.get("DARSHJ_WHISPER_MODEL", "base")
SERVER_PORT = int(os.environ.get("DARSHJ_PORT", "7860"))
SHARE = os.environ.get("DARSHJ_SHARE", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Fun loading messages & brand colours
# ---------------------------------------------------------------------------
LOADING_MESSAGES = [
    "🎯 Warming up the AI engines...",
    "🚀 Preparing transcription rockets...",
    "✨ Sprinkling some AI magic dust...",
    "🎙️ Tuning the digital microphones...",
    "🤖 Waking up the Whisper wizards...",
    "💫 Aligning the neural networks...",
    "🎬 Setting up the media processors...",
    "🔮 Consulting the transcription oracle...",
    "⚡ Charging the AI batteries...",
    "🎪 Starting the transcription show...",
]

BRAND_COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "accent": "#f093fb",
    "success": "#4ECDC4",
    "warning": "#FFE66D",
    "danger": "#FF6B6B",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DarshjStats:
    """Statistics tracker for Darshj.AI"""
    total_files: int = 0
    total_duration: float = 0.0
    total_words: int = 0
    total_characters: int = 0
    languages_detected: Dict[str, int] = field(default_factory=dict)
    file_formats: Dict[str, int] = field(default_factory=dict)
    processing_speeds: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    daily_usage: Dict[str, int] = field(default_factory=dict)

    def add_transcription(self, duration: float, text: str, language: str,
                          fmt: str, speed: float, confidence: float):
        self.total_files += 1
        self.total_duration += duration
        self.total_words += len(text.split())
        self.total_characters += len(text)
        self.languages_detected[language] = self.languages_detected.get(language, 0) + 1
        self.file_formats[fmt] = self.file_formats.get(fmt, 0) + 1
        self.processing_speeds.append(speed)
        self.confidence_scores.append(confidence)
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_usage[today] = self.daily_usage.get(today, 0) + 1

    def get_achievement_level(self) -> Tuple[str, str]:
        if self.total_files >= 100:
            return "🏆", "Transcription Master"
        elif self.total_files >= 50:
            return "🥇", "Transcription Expert"
        elif self.total_files >= 20:
            return "🥈", "Transcription Pro"
        elif self.total_files >= 10:
            return "🥉", "Transcription Enthusiast"
        elif self.total_files >= 5:
            return "⭐", "Rising Star"
        return "🌟", "Beginner"


@dataclass
class TranscriptionMemory:
    """Memory system for Darshj.AI"""
    transcriptions: List[Dict[str, Any]] = field(default_factory=list)
    max_memory: int = 50

    def add(self, text: str, metadata: Dict) -> str:
        entry = {
            "text": text,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "id": f"darshj_{len(self.transcriptions) + 1}",
        }
        self.transcriptions.append(entry)
        if len(self.transcriptions) > self.max_memory:
            self.transcriptions.pop(0)
        return entry["id"]

    def search(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        return [t for t in self.transcriptions if query_lower in t["text"].lower()]

    def get_recent(self, n: int = 5) -> List[Dict]:
        return self.transcriptions[-n:] if self.transcriptions else []


# ---------------------------------------------------------------------------
# Whisper engine
# ---------------------------------------------------------------------------

class DarshjWhisperWizard:
    """The magical Whisper transcription engine for Darshj.AI"""

    def __init__(self, model_size: str = "base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🧙‍♂️ Darshj.AI Whisper Wizard initializing...")
        print(f"   📍 Device: {self.device}")
        print(f"   📏 Model: {model_size}")
        print(f"   ⏳ Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"   ✅ Whisper Wizard ready for magic!\n")
        self.supported_formats = [
            ".mp3", ".wav", ".mp4", ".avi", ".mov", ".flac", ".m4a", ".webm", ".ogg"
        ]

    def extract_audio(self, input_path: str, progress_callback=None) -> str:
        if progress_callback:
            progress_callback("🎬 Extracting audio track...")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, tmp.name, acodec="pcm_s16le", ar="16000", ac=1)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            if progress_callback:
                progress_callback("✅ Audio extracted successfully!")
            return tmp.name
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Audio extraction failed: {e}")
            raise

    def transcribe(self, file_path: str, language: Optional[str] = None,
                   task: str = "transcribe", progress_callback=None) -> Dict:
        start_time = time.time()
        if progress_callback:
            progress_callback(random.choice(LOADING_MESSAGES))

        file_ext = Path(file_path).suffix.lower()
        audio_path = file_path

        if file_ext in [".mp4", ".avi", ".mov", ".webm"]:
            audio_path = self.extract_audio(file_path, progress_callback)

        try:
            if progress_callback:
                progress_callback("🤖 AI is listening to your file...")

            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                verbose=False,
                fp16=(self.device == "cuda"),
            )

            if progress_callback:
                progress_callback("📝 Converting speech to text...")

            duration = (
                result.get("segments", [{}])[-1].get("end", 0)
                if result.get("segments")
                else 0
            )
            process_time = time.time() - start_time
            speed_ratio = duration / process_time if process_time > 0 else 0
            word_count = len(result["text"].split())

            segments = []
            total_confidence = 0
            for seg in result.get("segments", []):
                confidence = np.exp(seg.get("avg_logprob", -1))
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "confidence": confidence,
                })
                total_confidence += confidence

            avg_confidence = total_confidence / len(segments) if segments else 0.5

            if audio_path != file_path:
                os.unlink(audio_path)

            if progress_callback:
                progress_callback(f"✨ Transcription complete! ({word_count} words)")

            return {
                "success": True,
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": segments,
                "duration": duration,
                "word_count": word_count,
                "process_time": process_time,
                "speed_ratio": speed_ratio,
                "confidence": avg_confidence,
                "format": file_ext,
            }
        except Exception as e:
            if audio_path != file_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class DarshjAnalytics:
    def __init__(self, stats: DarshjStats):
        self.stats = stats

    def create_fun_stats_card(self) -> str:
        if self.stats.total_files == 0:
            return (
                "<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                "padding: 30px; border-radius: 20px; color: white; text-align: center;'>"
                "<h2>🎯 Welcome to Darshj.AI!</h2>"
                "<p style='font-size: 1.2em;'>Upload your first file to start the magic! ✨</p>"
                "<div style='font-size: 3em; margin: 20px;'>🚀</div></div>"
            )

        emoji, level = self.stats.get_achievement_level()
        avg_speed = np.mean(self.stats.processing_speeds) if self.stats.processing_speeds else 0
        avg_confidence = np.mean(self.stats.confidence_scores) if self.stats.confidence_scores else 0

        return f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px; border-radius: 20px; color: white; position: relative;'>
            <div style='position: absolute; top: 20px; right: 20px; font-size: 2.5em;'>{emoji}</div>
            <h2 style='margin-top: 0;'>🎯 Darshj.AI Stats Dashboard</h2>
            <p style='opacity: 0.9; margin-bottom: 20px;'>Level: {level}</p>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;'>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
                    <div style='font-size: 2em; font-weight: bold;'>{self.stats.total_files}</div>
                    <div style='opacity: 0.8;'>Files Transcribed</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
                    <div style='font-size: 2em; font-weight: bold;'>{self.stats.total_words:,}</div>
                    <div style='opacity: 0.8;'>Words Processed</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
                    <div style='font-size: 2em; font-weight: bold;'>{humanize.naturaldelta(self.stats.total_duration)}</div>
                    <div style='opacity: 0.8;'>Audio Processed</div>
                </div>
            </div>
            <div style='margin-top: 20px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
                <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px;'>
                    <span style='opacity: 0.7;'>⚡ Speed:</span> {avg_speed:.1f}x realtime
                </div>
                <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px;'>
                    <span style='opacity: 0.7;'>🎯 Accuracy:</span> {avg_confidence:.1%}
                </div>
            </div>
            <div style='margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1);
                        border-radius: 10px; text-align: center;'>
                <span style='font-size: 1.5em;'>🏆</span>
                <span style='margin-left: 10px;'>
                    {100 - self.stats.total_files if self.stats.total_files < 100 else 0} files to next level!
                </span>
            </div>
        </div>
        """

    def create_language_donut(self) -> go.Figure:
        if not self.stats.languages_detected:
            fig = go.Figure()
            fig.add_annotation(text="🎯 No data yet", showarrow=False, font=dict(size=20))
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig

        lang_emojis = {
            "en": "🇬🇧", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪",
            "it": "🇮🇹", "pt": "🇵🇹", "ru": "🇷🇺", "ja": "🇯🇵",
            "ko": "🇰🇷", "zh": "🇨🇳", "hi": "🇮🇳", "ne": "🇳🇵", "gu": "🇮🇳",
        }
        labels = [f"{lang_emojis.get(k, '🌍')} {k.upper()}" for k in self.stats.languages_detected]
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=list(self.stats.languages_detected.values()),
            hole=0.4,
            marker=dict(colors=px.colors.sequential.Plasma, line=dict(color="white", width=2)),
            textfont=dict(size=14, color="white"),
            hovertemplate="<b>%{label}</b><br>Files: %{value}<br>%{percent}<extra></extra>",
        )])
        fig.add_annotation(text="🌍", showarrow=False, font=dict(size=40))
        fig.update_layout(
            title="<b>🗣️ Language Distribution</b>", height=350, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#667eea"),
        )
        return fig

    def create_speed_chart(self) -> go.Figure:
        if not self.stats.processing_speeds:
            fig = go.Figure()
            fig.add_annotation(text="⚡ No speed data yet", showarrow=False, font=dict(size=20))
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig

        x_data = list(range(1, len(self.stats.processing_speeds) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data, y=self.stats.processing_speeds, mode="lines+markers", name="Speed",
            line=dict(color="#667eea", width=3), marker=dict(size=10, color="#764ba2"),
            fill="tozeroy", fillcolor="rgba(102, 126, 234, 0.2)",
            hovertemplate="File #%{x}<br>Speed: %{y:.1f}x<extra></extra>",
        ))
        avg_speed = np.mean(self.stats.processing_speeds)
        fig.add_hline(y=avg_speed, line_dash="dash", line_color="#FF6B6B",
                      annotation_text=f"Avg: {avg_speed:.1f}x")
        fig.update_layout(
            title="<b>⚡ Processing Speed (Realtime Multiple)</b>",
            xaxis_title="File Number", yaxis_title="Speed (x realtime)", height=350,
            hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#667eea"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.1)"), yaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
        )
        return fig

    def create_daily_heatmap(self) -> go.Figure:
        if not self.stats.daily_usage:
            fig = go.Figure()
            fig.add_annotation(text="📅 No daily data yet", showarrow=False, font=dict(size=20))
            fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig

        dates = pd.date_range(end=datetime.now(), periods=30)
        values = [self.stats.daily_usage.get(d.strftime("%Y-%m-%d"), 0) for d in dates]
        fig = go.Figure(data=go.Heatmap(
            z=[values[i : i + 7] for i in range(0, len(values), 7)],
            colorscale="Plasma", showscale=False,
            hovertemplate="Files: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title="<b>📅 30-Day Activity</b>", height=200,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#667eea"),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        return fig


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class DarshjTranscriptionDashboard:
    def __init__(self):
        print("\n🎯 Initializing Darshj.AI Dashboard...")
        self.wizard = DarshjWhisperWizard(model_size=WHISPER_MODEL)
        self.stats = DarshjStats()
        self.memory = TranscriptionMemory()
        self.analytics = DarshjAnalytics(self.stats)
        print("✨ Darshj.AI Dashboard ready!\n")

    def process_file(self, file, language, task, export_format):
        if file is None:
            return (
                "🎯 Please upload a file to start the magic!",
                "",
                self.analytics.create_fun_stats_card(),
                self.analytics.create_language_donut(),
                self.analytics.create_speed_chart(),
                None,
            )

        progress_messages: List[str] = []

        result = self.wizard.transcribe(
            file.name,
            language=language if language != "Auto-detect" else None,
            task=task,
            progress_callback=progress_messages.append,
        )

        if not result["success"]:
            return (
                f"❌ Oops! Something went wrong: {result.get('error', 'Unknown error')}",
                "",
                self.analytics.create_fun_stats_card(),
                self.analytics.create_language_donut(),
                self.analytics.create_speed_chart(),
                None,
            )

        self.stats.add_transcription(
            duration=result["duration"],
            text=result["text"],
            language=result["language"],
            fmt=result["format"],
            speed=result["speed_ratio"],
            confidence=result["confidence"],
        )

        trans_id = self.memory.add(result["text"], {
            "language": result["language"],
            "duration": result["duration"],
            "word_count": result["word_count"],
            "confidence": result["confidence"],
        })

        output_text = (
            f"🎯 **Darshj.AI Transcription Complete!**\n\n"
            f"📊 **Stats:**\n"
            f"• Language: {result['language'].upper()}\n"
            f"• Duration: {humanize.naturaldelta(result['duration'])}\n"
            f"• Words: {result['word_count']:,}\n"
            f"• Speed: {result['speed_ratio']:.1f}x realtime\n"
            f"• Confidence: {result['confidence']:.1%}\n"
            f"• ID: {trans_id}\n\n"
            f"📝 **Transcription:**\n{result['text']}"
        )

        progress_log = "\n".join([f"• {msg}" for msg in progress_messages])

        export_file = None
        if export_format != "None":
            export_file = self._create_export(result["text"], export_format, trans_id)

        return (
            output_text,
            progress_log,
            self.analytics.create_fun_stats_card(),
            self.analytics.create_language_donut(),
            self.analytics.create_speed_chart(),
            export_file,
        )

    @staticmethod
    def _create_export(text: str, fmt: str, trans_id: str) -> Optional[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"darshjai_transcription_{timestamp}"

        if fmt == "Text (.txt)":
            filepath = f"{filename}.txt"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("Darshj.AI Transcription\n")
                f.write(f"ID: {trans_id}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(text)
            return filepath
        elif fmt == "JSON (.json)":
            filepath = f"{filename}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "transcription_id": trans_id,
                    "timestamp": datetime.now().isoformat(),
                    "text": text,
                    "word_count": len(text.split()),
                    "character_count": len(text),
                    "powered_by": "Darshj.AI",
                }, f, indent=2)
            return filepath
        return None

    def search_memory(self, query: str) -> str:
        if not query:
            return "🔍 Enter a search query to find past transcriptions"
        results = self.memory.search(query)
        if not results:
            return f"❌ No results found for '{query}'"
        output = f"🔍 **Found {len(results)} result(s) for '{query}':**\n\n"
        for i, r in enumerate(results, 1):
            output += f"**Result {i}** (ID: {r['id']})\nDate: {r['timestamp']}\nPreview: {r['text'][:200]}...\n\n"
        return output

    def build_interface(self) -> gr.Blocks:
        custom_css = """
        .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; }
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
        }
        .gr-button-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        }
        """

        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo"),
            css=custom_css,
            title="🎯 Darshj.AI - Transcription Magic",
        ) as interface:

            gr.HTML("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 40px; border-radius: 20px; text-align: center; color: white;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
                <h1 style="margin: 0; font-size: 3.5em; font-weight: bold;">🎯 Darshj.AI</h1>
                <p style="font-size: 1.3em; margin-top: 10px; opacity: 0.95;">
                    ✨ Transcribe Anything, Anywhere, Anytime! ✨</p>
                <div style="margin-top: 20px;">
                    <span style="background: rgba(255,255,255,0.2); padding: 5px 15px;
                                 border-radius: 20px; margin: 0 5px;">🎙️ Audio</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 5px 15px;
                                 border-radius: 20px; margin: 0 5px;">🎬 Video</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 5px 15px;
                                 border-radius: 20px; margin: 0 5px;">🤖 AI-Powered</span>
                </div>
            </div>
            """)

            with gr.Tabs():
                # ── Transcribe Tab ──
                with gr.TabItem("🎙️ Transcribe", id=1):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📤 Upload Your File")
                            file_input = gr.File(
                                label="Drag & Drop or Click to Upload",
                                file_types=[".mp3", ".wav", ".mp4", ".avi", ".mov",
                                            ".flac", ".m4a", ".webm", ".ogg"],
                            )
                            language = gr.Dropdown(
                                choices=[
                                    "Auto-detect",
                                    "en", "es", "fr", "de", "it", "pt", "ru",
                                    "ja", "ko", "zh",
                                    "hi", "ne", "gu",
                                ],
                                value="Auto-detect",
                                label="🌍 Language",
                                interactive=True,
                            )
                            task = gr.Radio(
                                choices=["transcribe", "translate"],
                                value="transcribe",
                                label="📝 Task",
                                info="Transcribe keeps original language, Translate converts to English",
                            )
                            export_format = gr.Radio(
                                choices=["None", "Text (.txt)", "JSON (.json)"],
                                value="None",
                                label="💾 Export Format",
                            )
                            process_btn = gr.Button("🚀 Start Transcription Magic!", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            gr.Markdown("### 📝 Results")
                            output_text = gr.Textbox(label="Transcription Output", lines=15, max_lines=25,
                                                     placeholder="Your transcription will appear here... ✨")
                            progress_log = gr.Textbox(label="🎬 Progress Log", lines=3,
                                                      placeholder="Processing steps will appear here...")
                            export_file = gr.File(label="📥 Download Export", visible=True)

                # ── Analytics Tab ──
                with gr.TabItem("📊 Analytics", id=2):
                    gr.Markdown("### 📈 Your Transcription Analytics")
                    stats_display = gr.HTML(value=self.analytics.create_fun_stats_card())
                    with gr.Row():
                        lang_chart = gr.Plot(value=self.analytics.create_language_donut())
                        speed_chart = gr.Plot(value=self.analytics.create_speed_chart())
                    activity_chart = gr.Plot(value=self.analytics.create_daily_heatmap())
                    refresh_btn = gr.Button("🔄 Refresh Analytics", size="sm")

                # ── Memory Tab ──
                with gr.TabItem("🧠 Memory", id=3):
                    gr.Markdown("### 🔍 Search Your Transcription History")
                    search_input = gr.Textbox(label="Search Query",
                                              placeholder="Enter keywords to search past transcriptions...")
                    search_btn = gr.Button("🔍 Search", variant="primary")
                    search_results = gr.Textbox(label="Search Results", lines=10,
                                                placeholder="Results will appear here...")
                    gr.Markdown("### 📜 Recent Transcriptions")
                    recent_btn = gr.Button("📜 Show Recent", size="sm")
                    recent_display = gr.JSON(label="Recent Transcriptions")

                # ── About Tab ──
                with gr.TabItem("ℹ️ About", id=4):
                    gr.Markdown("""
## 🎯 About Darshj.AI

### 🚀 Features
- **🎙️ Universal Support**: Transcribe any audio/video format
- **🤖 AI-Powered**: Using OpenAI's Whisper technology
- **⚡ Lightning Fast**: GPU-accelerated processing
- **🌍 Multilingual**: Support for 13+ languages including Hindi, Nepali & Gujarati
- **📊 Analytics**: Track your usage and performance
- **🧠 Memory**: Search through past transcriptions
- **💾 Export**: Save in multiple formats

### 🏆 Achievement Levels
- 🌟 **Beginner** (0-4 files)
- ⭐ **Rising Star** (5-9 files)
- 🥉 **Enthusiast** (10-19 files)
- 🥈 **Pro** (20-49 files)
- 🥇 **Expert** (50-99 files)
- 🏆 **Master** (100+ files)

### 👨‍💻 Created by Darshj.AI
Made with ❤️ and lots of ☕
                    """)

            gr.HTML("""
            <div style="text-align: center; padding: 20px; opacity: 0.7;">
                <p>Powered by 🎯 Darshj.AI | Built with Whisper & Gradio |
                <span style="color: #667eea;">v2.0.0</span></p>
            </div>
            """)

            # ── Event handlers ──
            process_btn.click(
                fn=self.process_file,
                inputs=[file_input, language, task, export_format],
                outputs=[output_text, progress_log, stats_display, lang_chart, speed_chart, export_file],
            )
            refresh_btn.click(
                fn=lambda: [
                    self.analytics.create_fun_stats_card(),
                    self.analytics.create_language_donut(),
                    self.analytics.create_speed_chart(),
                    self.analytics.create_daily_heatmap(),
                ],
                outputs=[stats_display, lang_chart, speed_chart, activity_chart],
            )
            search_btn.click(fn=self.search_memory, inputs=[search_input], outputs=[search_results])
            recent_btn.click(fn=lambda: self.memory.get_recent(5), outputs=[recent_display])

        return interface


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("🎯 DARSHJ.AI TRANSCRIPTION DASHBOARD")
    print("=" * 60)
    print(f"🔥 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"💻 System: {platform.system()} {platform.machine()}")
    print(f"📏 Whisper model: {WHISPER_MODEL}")

    dashboard = DarshjTranscriptionDashboard()
    interface = dashboard.build_interface()

    print(f"\n🌟 Launching Darshj.AI Dashboard on port {SERVER_PORT}...")
    interface.launch(
        share=SHARE,
        server_name="0.0.0.0",
        server_port=SERVER_PORT,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
