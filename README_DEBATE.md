# Philosophical Debate System - Quick Start

## 🎭 What is This?

Two Grok LLM instances debate deep philosophical questions with **different voices** using Qwen3 TTS!

- **The Rationalist** 🧠: Analytical, deep contemplative voice
- **The Phenomenologist** 💭: Intuitive, warm energetic voice

---

## 🚀 Access the App

**Web Interface**: https://paarijatthakur--philosophical-debate-web.modal.run

1. Choose a philosophical question
2. Set number of exchanges (default: 6)
3. Click "Start Philosophical Debate"
4. Wait 2-3 minutes
5. Listen to the AI debate with distinct voices!

---

## 📝 Available Topics

1. **Consciousness & AI**: Can AI truly achieve consciousness?
2. **Free Will**: Is free will real or an illusion?
3. **AI Ethics**: How should we guide superintelligent AI?
4. **Meaning & Purpose**: Is meaning inherent or created?
5. **Epistemology**: Can we know anything with certainty?

---

## 🛠️ Local Testing (Optional)

```bash
# Activate environment
source serverless/bin/activate

# Run a debate
modal run philosophical_debate_modal.py --question-id 1 --num-exchanges 6
```

This saves audio as: `debate_1_YYYYMMDD_HHMMSS.wav`

---

## 📂 Files

- **philosophical_debate_modal.py**: Complete Modal app
- **Modal secret**: `grok-api-key` (already configured)

---

Enjoy the philosophical discourse! 🎭✨
