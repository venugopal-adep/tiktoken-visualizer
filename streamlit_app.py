import html
import json
import re
import pandas as pd
import streamlit as st
import tiktoken

# Page Configuration
st.set_page_config(
    page_title="TikTokenizer Studio 🔠",
    page_icon="🔠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Supported Encodings
ENCODING_INFO = {
    "o200k_base": {
        "label": "o200k_base (GPT-4o, GPT-4o mini)",
        "models": "GPT-4o, GPT-4o mini",
        "vocab_size": "200,019",
        "description": "OpenAI's latest state-of-the-art tokenizer with 200k vocabulary. Drastically improves compression on non-English text, code, and math.",
        "cost_input_per_m": 2.50,
    },
    "cl100k_base": {
        "label": "cl100k_base (GPT-4, GPT-3.5 Turbo)",
        "models": "GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-ada-002, text-embedding-3",
        "vocab_size": "100,277",
        "description": "Standard workhorse tokenizer for GPT-4 and GPT-3.5 Turbo with 100k vocabulary.",
        "cost_input_per_m": 10.00,
    },
    "p50k_base": {
        "label": "p50k_base (Codex, text-davinci-002/003)",
        "models": "Codex, text-davinci-002, text-davinci-003",
        "vocab_size": "50,281",
        "description": "Optimized for programming languages and source code.",
        "cost_input_per_m": 20.00,
    },
    "p50k_edit": {
        "label": "p50k_edit (text-davinci-edit-001)",
        "models": "text-davinci-edit-001, code-davinci-edit-001",
        "vocab_size": "50,284",
        "description": "Specialized tokenizer for OpenAI edit and insertion endpoints.",
        "cost_input_per_m": 20.00,
    },
    "r50k_base": {
        "label": "r50k_base (GPT-3 Legacy)",
        "models": "GPT-3 (davinci, curie, babbage, ada)",
        "vocab_size": "50,257",
        "description": "Original GPT-3 50k vocabulary tokenizer.",
        "cost_input_per_m": 20.00,
    },
    "gpt2": {
        "label": "gpt2 (GPT-2 Foundational)",
        "models": "GPT-2",
        "vocab_size": "50,257",
        "description": "The foundational BPE tokenizer from the 2019 GPT-2 paper.",
        "cost_input_per_m": 0.00,
    },
}

COLOR_PALETTE = [
    {"bg": "#ede9fe", "border": "#8b5cf6", "text": "#5b21b6"},  # Violet
    {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e40af"},  # Blue
    {"bg": "#dcfce7", "border": "#22c55e", "text": "#166534"},  # Green
    {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e"},  # Amber
    {"bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b"},  # Red
    {"bg": "#fce7f3", "border": "#ec4899", "text": "#9d174d"},  # Pink
    {"bg": "#ccfbf1", "border": "#14b8a6", "text": "#115e59"},  # Teal
    {"bg": "#ffedd5", "border": "#f97316", "text": "#9a3412"},  # Orange
    {"bg": "#e0e7ff", "border": "#6366f1", "text": "#3730a3"},  # Indigo
    {"bg": "#ecfccb", "border": "#84cc16", "text": "#3f6212"},  # Lime
    {"bg": "#cffafe", "border": "#06b6d4", "text": "#155e75"},  # Cyan
    {"bg": "#f3e8ff", "border": "#a855f7", "text": "#6b21a8"},  # Purple
]

PRESETS = {
    "👋 Hello World & Punctuation": "Hello, world! Welcome to the interactive Tiktokenizer visualizer. How does Byte Pair Encoding work?",
    "🐍 Python Code & Whitespace": """def fibonacci(n: int) -> list[int]:
    \"\"\"Return first n Fibonacci numbers.\"\"\"
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]""",
    "🌍 Multilingual & Emojis": "OpenAI GPT-4o supports multilingual tokenization! 🚀\nहिंदी: नमस्ते दुनिया, आप कैसे हैं?\n日本語: こんにちは世界！\nالعربية: مرحباً بالعالم\nFrançais: Bonjour le monde! 🌍✨",
    "🔢 Numbers & Math Expressions": "Calculate: 1234567890 + 9876543210 = 11111111100\nPi: 3.141592653589793\nPrice: $49.99 (Discount: 15% off) -> Total: $42.49",
    "🔒 Special Tokens": "<|endoftext|>\n<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>",
}

# Custom CSS for Tiktokenizer aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

.hero-box {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 40%, #8b5cf6 70%, #ec4899 100%);
    border-radius: 14px;
    padding: 22px 28px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.25);
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 4px 0;
    color: white !important;
}

.hero-sub {
    font-size: 14px;
    opacity: 0.94;
    margin: 0;
}

.tokens-container {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 16px;
    min-height: 120px;
    max-height: 480px;
    overflow-y: auto;
    line-height: 2.2;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    white-space: pre-wrap;
    word-break: break-word;
    margin-bottom: 16px;
}

.tok-chip {
    display: inline;
    padding: 3px 6px;
    margin: 1.5px;
    border-radius: 5px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}

.whitespace-glyph {
    opacity: 0.45;
    font-size: 12px;
}

.newline-glyph {
    opacity: 0.65;
    font-size: 12px;
    color: #ef4444;
}

.compare-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
}

.compare-title {
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #4f46e5;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def get_cached_encoding(name: str):
    return tiktoken.get_encoding(name)

def format_token_text(raw_bytes: bytes, show_whitespace: bool) -> tuple[str, str]:
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("utf-8", errors="replace")
    
    plain = decoded
    if show_whitespace:
        out = []
        for ch in decoded:
            if ch == " ":
                out.append('<span class="whitespace-glyph">·</span>')
            elif ch == "\n":
                out.append('<span class="newline-glyph">↵</span><br/>')
            elif ch == "\t":
                out.append('<span class="whitespace-glyph">⇥   </span>')
            else:
                out.append(html.escape(ch))
        return "".join(out), plain
    else:
        return html.escape(decoded).replace("\n", "<br/>"), plain

def render_chips_html(tokens: list[int], enc, show_whitespace: bool) -> str:
    if not tokens:
        return '<div style="color: #94a3b8; font-style: italic;">Enter text above to see tokens rendered.</div>'
    
    chips = []
    for idx, t in enumerate(tokens):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        try:
            b = enc.decode_single_token_bytes(t)
        except Exception:
            b = b"<err>"
        
        display_str, plain_str = format_token_text(b, show_whitespace)
        hex_id = f"0x{t:04X}"
        tooltip = f"Token #{idx} | ID: {t} ({hex_id}) | Bytes: {len(b)} | String: {repr(plain_str)}"
        style = f"background-color: {color['bg']}; border: 1px solid {color['border']}; color: {color['text']};"
        chips.append(f'<span class="tok-chip" title="{html.escape(tooltip)}" style="{style}">{display_str}</span>')
    
    return f'<div class="tokens-container">{"".join(chips)}</div>'

# Hero Banner
st.markdown("""
<div class="hero-box">
    <div class="hero-title">TikTokenizer Studio 🔠</div>
    <div class="hero-sub">Interactive Byte Pair Encoding (BPE) Visualizer & Multi-Model Token Inspector for OpenAI Models</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
selected_model = st.sidebar.selectbox(
    "Tokenizer Encoding",
    options=list(ENCODING_INFO.keys()),
    index=0,
    format_func=lambda k: ENCODING_INFO[k]["label"],
)

info = ENCODING_INFO[selected_model]
st.sidebar.caption(f"**Vocab Size**: {info['vocab_size']} | **Models**: {info['models']}")

st.sidebar.divider()
selected_preset = st.sidebar.selectbox("Load Example Snippet", options=["-- None --"] + list(PRESETS.keys()))

st.sidebar.divider()
show_ws = st.sidebar.checkbox("Show whitespace glyphs (·, ↵, ⇥)", value=True)
allow_special = st.sidebar.checkbox("Allow Special Tokens (e.g. <|endoftext|>)", value=True)

# Main Navigation Tabs
tab_encode, tab_decode, tab_compare, tab_guide = st.tabs([
    "🔍 Visual Tokenizer (Encode)",
    "🔄 Token Decoder (Decode)",
    "⚖️ Multi-Model Comparison",
    "📚 BPE Deep Dive",
])

# Default text handling
default_text = "Hello, world! 🌍 Welcome to TikTokenizer Studio.\nTry typing or pasting any text, code, or emojis to visualize the tokens in real-time."
if selected_preset != "-- None --" and selected_preset in PRESETS:
    default_text = PRESETS[selected_preset]

# TAB 1: ENCODE
with tab_encode:
    user_text = st.text_area("Input Text", value=default_text, height=130, placeholder="Type or paste text here...")
    
    enc = get_cached_encoding(selected_model)
    if user_text:
        try:
            if allow_special:
                tokens = enc.encode(user_text, allowed_special="all")
            else:
                tokens = enc.encode(user_text, disallowed_special=())
        except Exception as e:
            st.error(f"Tokenization error: {e}")
            tokens = []
    else:
        tokens = []

    # Metrics
    tok_count = len(tokens)
    char_count = len(user_text)
    byte_count = len(user_text.encode("utf-8"))
    ratio = (char_count / tok_count) if tok_count > 0 else 0.0
    cost_per_m = info.get("cost_input_per_m", 2.50)
    est_cost = (tok_count / 1_000_000.0) * cost_per_m

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tokens", f"{tok_count:,}")
    m2.metric("Characters", f"{char_count:,}")
    m3.metric("Chars / Token", f"{ratio:.2f}")
    m4.metric("Bytes (UTF-8)", f"{byte_count:,}")
    m5.metric("Est. Cost", f"${est_cost:.6f}")

    st.subheader("Color-Coded Token Stream")
    chips_html = render_chips_html(tokens, enc, show_ws)
    st.markdown(chips_html, unsafe_allow_html=True)
    st.caption("💡 Hover over any token chip to view its Token ID, byte length, and raw string.")

    with st.expander("📋 Token IDs & Sequence Formats", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.text_area("Python / JSON Array", value=json.dumps(tokens), height=80)
        with c2:
            st.text_area("Space-Separated IDs", value=" ".join(str(t) for t in tokens), height=80)

    with st.expander("📊 Detailed Token Inspector Table", expanded=False):
        if tokens:
            rows = []
            for idx, t in enumerate(tokens):
                try:
                    b = enc.decode_single_token_bytes(t)
                except Exception:
                    b = b"<err>"
                try:
                    s = b.decode("utf-8")
                except UnicodeDecodeError:
                    s = b.decode("utf-8", errors="replace")
                
                rows.append({
                    "Index": f"#{idx}",
                    "Token ID": t,
                    "Hex": f"0x{t:04X}",
                    "Decoded String": repr(s)[1:-1],
                    "Bytes (Hex)": " ".join(f"0x{x:02X}" for x in b),
                    "Length": f"{len(b)}B",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# TAB 2: DECODE
with tab_decode:
    st.markdown("### Decode Token IDs back to Original Text")
    st.markdown("Enter integer token IDs (comma-separated, space-separated, or JSON list).")
    
    decode_model = st.selectbox(
        "Decoder Encoding",
        options=list(ENCODING_INFO.keys()),
        index=0,
        format_func=lambda k: ENCODING_INFO[k]["label"],
        key="decode_model_key"
    )
    decode_input = st.text_area("Token IDs Input", value="13225, 11, 2375, 0, 130321, 235", height=90)
    
    if decode_input.strip():
        nums = re.findall(r"\d+", decode_input)
        if nums:
            dec_tokens = [int(n) for n in nums]
            d_enc = get_cached_encoding(decode_model)
            try:
                decoded_str = d_enc.decode(dec_tokens)
                st.text_area("Decoded Plain Text", value=decoded_str, height=100)
                st.subheader("Reconstructed Token Visualizer")
                st.markdown(render_chips_html(dec_tokens, d_enc, show_ws), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Decoding failed: {e}")
        else:
            st.warning("No valid integer token IDs found in input.")

# TAB 3: MULTI-MODEL COMPARISON
with tab_compare:
    st.markdown("### Compare BPE Tokenizers Side-by-Side")
    st.markdown("See how different generations of tokenizers encode the exact same text:")
    
    comp_text = st.text_area(
        "Comparison Text",
        value="Artificial Intelligence is revolutionizing natural language processing across multiple languages: हिंदी, 日本語, and Français! 🚀",
        height=90,
    )

    if comp_text.strip():
        models_to_compare = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base", "gpt2"]
        comp_rows = []
        base_tokens = len(get_cached_encoding("o200k_base").encode(comp_text))

        for m in models_to_compare:
            m_enc = get_cached_encoding(m)
            m_tokens = m_enc.encode(comp_text)
            cnt = len(m_tokens)
            chars_tok = (len(comp_text) / cnt) if cnt > 0 else 0.0
            diff = "Baseline (0%)" if cnt == base_tokens else f"+{(((cnt - base_tokens) / base_tokens) * 100):.1f}% more tokens"
            
            comp_rows.append({
                "Encoding": m,
                "Target Models": ENCODING_INFO[m]["models"],
                "Vocab Size": ENCODING_INFO[m]["vocab_size"],
                "Tokens": cnt,
                "Chars/Token": round(chars_tok, 2),
                "Efficiency vs o200k": diff,
            })
            
            with st.container():
                st.markdown(f"""
                <div class="compare-card">
                    <span class="compare-title">{m}</span> &bull; 
                    <strong>{cnt}</strong> tokens &bull; {chars_tok:.2f} chars/tok &bull; 
                    <span style="color: #4f46e5; font-weight: 600;">{diff}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(render_chips_html(m_tokens, m_enc, show_ws), unsafe_allow_html=True)
        
        st.subheader("Summary Table")
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

# TAB 4: GUIDE
with tab_guide:
    st.markdown("""
    ## Understanding Byte Pair Encoding (BPE) & Tiktoken

    ### 1. What is Byte Pair Encoding?
    Language models don't read words or letters directly—they operate on integer **Tokens**. 
    BPE begins with raw bytes and iteratively merges the most frequently co-occurring byte pairs in the training corpus into single tokens.

    ### 2. Why does leading whitespace matter?
    In modern tokenizers (`cl100k_base`, `o200k_base`), spaces are usually fused to the *beginning* of words:
    * `" hello"` (with space) is a single token (`24748` in o200k).
    * `"hello"` (without space) is a distinct token (`40617` in o200k).
    This is why prompt indentation, trailing spaces, or missing spaces alter token boundaries.

    ### 3. The Evolution of OpenAI Tokenizers
    * **`gpt2` / `r50k_base`** (50,257 vocab): Foundational 50k vocabulary. Inefficient on code and non-English text.
    * **`p50k_base`** (50,281 vocab): Tuned for Codex and programming languages.
    * **`cl100k_base`** (100,277 vocab): Standard for GPT-4 / GPT-3.5. 2x vocabulary.
    * **`o200k_base`** (200,019 vocab): Latest 200k vocabulary for GPT-4o. Up to **40% token reduction** on languages like Hindi, Japanese, Arabic, and Python!
    """)
