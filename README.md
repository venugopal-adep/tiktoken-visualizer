# TikTokenizer Studio 🔠

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=venugopal-adep/tiktoken-visualizer&branch=main&main_module=streamlit_app.py)

An interactive Byte Pair Encoding (BPE) visualizer and multi-model token inspector inspired by [Tiktokenizer](https://tiktokenizer.vercel.app/), built with **Streamlit** and **Gradio**.

## 🚀 One-Click Deploy to Streamlit Community Cloud

Click the button above or use this link:
**[Deploy to Streamlit Cloud](https://share.streamlit.io/deploy?repository=venugopal-adep/tiktoken-visualizer&branch=main&main_module=streamlit_app.py)**

---

## 🌟 Key Features

1. **🎨 Color-Coded Token Stream**:
   - Every token is rendered with alternating vibrant pastel badges.
   - Visible whitespace glyphs (`·` for spaces, `↵` for linebreaks, `⇥` for tabs).
   - Real-time hover inspector displaying Token ID, Hex code, UTF-8 byte representation, and decoded string.

2. **⚡ Multiple OpenAI Encodings Supported**:
   - `o200k_base` (GPT-4o, GPT-4o mini - 200k vocab)
   - `cl100k_base` (GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, Ada-002 - 100k vocab)
   - `p50k_base` (Codex, text-davinci-002 / text-davinci-003 - 50k vocab)
   - `p50k_edit` (Edit / Insertion endpoints)
   - `r50k_base` (Legacy GPT-3 davinci, curie, babbage, ada)
   - `gpt2` (Foundational BPE GPT-2)

3. **📊 Live Metrics & Cost Estimation**:
   - Real-time count of total tokens, raw characters, and bytes.
   - Compression efficiency ratio (Characters / Token).
   - Cost projection for input tokens per 1M tokens.

4. **🔄 Bidirectional Pipeline**:
   - **Encoder**: Text ➔ Color-coded Tokens, JSON array `[ID, ID, ...]`, Space-delimited list, and detailed token table.
   - **Decoder**: Token IDs ➔ Reconstructed text and visual chips.

5. **⚖️ Multi-Model Comparison Matrix**:
   - Compare `o200k_base` vs `cl100k_base` vs `p50k_base` vs `r50k_base` vs `gpt2` on the exact same prompt.
   - Displays compression percentage gain side-by-side with visual token previews.

6. **📚 Interactive Educational Guide**:
   - Explains BPE mechanics, whitespace tokenization rules, Unicode byte splits, and tokenizer evolution.

---

## 💻 Local Execution

### Run with Streamlit:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Run with Gradio:
```bash
python app.py
```
