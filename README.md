---
title: TikTokenizer Studio
emoji: 🔠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive Tiktoken BPE Visualizer with Gradio
---

# TikTokenizer Studio 🔠

An interactive Byte Pair Encoding (BPE) visualizer and multi-model token inspector inspired by [Tiktokenizer](https://tiktokenizer.vercel.app/), built with **Gradio 5**.

## 🌟 Features

* **🎨 Color-Coded Token Stream**: Tokens highlighted in an alternating 12-color pastel palette.
* **⚡ Multi-Encoding Support**: `o200k_base` (GPT-4o, GPT-4o mini), `cl100k_base` (GPT-4), `p50k_base` (Codex), `p50k_edit`, `r50k_base`, and `gpt2`.
* **🔍 Real-Time Inspector**: Hover over any token to inspect its Token ID, Hex, Length, Bytes, and String.
* **📊 Metrics Dashboard**: Real-time token counts, character counts, compression ratios (chars/token), and estimated input costs.
* **🔄 Bidirectional Pipeline**: Encode text to tokens & decode token IDs back to text.
* **⚖️ Multi-Model Comparison**: Side-by-side token efficiency comparison across OpenAI model generations.
* **📚 BPE Educational Deep Dive**: Explaining subword tokenization, leading spaces, and UTF-8 byte splitting.
