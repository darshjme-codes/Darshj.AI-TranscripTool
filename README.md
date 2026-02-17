# 🎯 Darshj.AI - Ultimate Transcription Dashboard

<div align="center">
    <img src="https://img.shields.io/badge/Darshj.AI-Transcription_Magic-FF6B6B?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Powered_by-Whisper_AI-4ECDC4?style=for-the-badge&logo=openai&logoColor=white" />
    <img src="https://img.shields.io/badge/Built_with-Gradio-FFE66D?style=for-the-badge&logo=gradio&logoColor=black" />
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</div>

<div align="center">
    <h3>✨ Transcribe Anything, Anywhere, Anytime! ✨</h3>
    <p>Transform your audio and video files into text with AI-powered magic!</p>
</div>

---

## 🚀 Features

- 🎙️ **Universal Media Support** — MP3, WAV, MP4, AVI, MOV, FLAC, M4A, WebM, OGG
- 🤖 **AI-Powered** — OpenAI Whisper transcription & translation
- 🌍 **13+ Languages** — English, Spanish, French, German, Hindi, Nepali, Gujarati & more
- 📊 **Analytics Dashboard** — Charts, heatmaps, achievement levels
- 🧠 **Smart Memory** — Search through past transcriptions
- 💾 **Export** — TXT & JSON with metadata
- 🎨 **Beautiful UI** — Gradio-based responsive dashboard

---

## 🛠️ Installation

### Option 1: Python (recommended)

```bash
git clone https://github.com/darshjme-codes/Darshj.AI-TranscripTool.git
cd Darshj.AI-TranscripTool
pip install -r requirements.txt
python app.py
```

> **Note:** FFmpeg must be installed on your system (`apt install ffmpeg` / `brew install ffmpeg`).

### Option 2: Docker

```bash
docker build -t darshjai-transcriptool .
docker run -p 7860:7860 darshjai-transcriptool
```

Then open **http://localhost:7860** in your browser.

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `DARSHJ_WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `DARSHJ_PORT` | `7860` | Server port |
| `DARSHJ_SHARE` | `false` | Set `true` to create a public Gradio link |

Example:

```bash
DARSHJ_WHISPER_MODEL=small DARSHJ_SHARE=true python app.py
```

---

## 🏆 Achievement Levels

| Files | Level |
|---|---|
| 0–4 | 🌟 Beginner |
| 5–9 | ⭐ Rising Star |
| 10–19 | 🥉 Enthusiast |
| 20–49 | 🥈 Pro |
| 50–99 | 🥇 Expert |
| 100+ | 🏆 Master |

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
    <h3>Made with ❤️ by Darshj.AI</h3>
</div>
