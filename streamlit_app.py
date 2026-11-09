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
        "description": "OpenAI's flagship 200k vocabulary tokenizer. Drastically improves compression on non-English text, code, and math.",
        "cost_input_per_m": 2.50,
        "badge_color": "#10b981",
    },
    "cl100k_base": {
        "label": "cl100k_base (GPT-4, GPT-3.5 Turbo)",
        "models": "GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, Ada-002",
        "vocab_size": "100,277",
        "description": "Standard workhorse tokenizer for GPT-4 and GPT-3.5 Turbo.",
        "cost_input_per_m": 10.00,
        "badge_color": "#6366f1",
    },
    "p50k_base": {
        "label": "p50k_base (Codex, text-davinci-002/003)",
        "models": "Codex, text-davinci-002, text-davinci-003",
        "vocab_size": "50,281",
        "description": "Optimized for programming languages and source code.",
        "cost_input_per_m": 20.00,
        "badge_color": "#f59e0b",
    },
    "p50k_edit": {
        "label": "p50k_edit (text-davinci-edit-001)",
        "models": "text-davinci-edit-001, code-davinci-edit-001",
        "vocab_size": "50,284",
        "description": "Specialized tokenizer for OpenAI edit endpoints.",
        "cost_input_per_m": 20.00,
        "badge_color": "#ec4899",
    },
    "r50k_base": {
        "label": "r50k_base (GPT-3 Legacy)",
        "models": "GPT-3 (davinci, curie, babbage, ada)",
        "vocab_size": "50,257",
        "description": "Original GPT-3 50k vocabulary tokenizer.",
        "cost_input_per_m": 20.00,
        "badge_color": "#8b5cf6",
    },
    "gpt2": {
        "label": "gpt2 (GPT-2 Foundational)",
        "models": "GPT-2",
        "vocab_size": "50,257",
        "description": "The foundational BPE tokenizer from the 2019 GPT-2 paper.",
        "cost_input_per_m": 0.00,
        "badge_color": "#06b6d4",
    },
}

# 16 High-Vibrancy Pastel/Neon Palette with High Contrast
COLOR_PALETTE = [
    {"bg": "#ede9fe", "border": "#8b5cf6", "text": "#4c1d95", "shadow": "rgba(139, 92, 246, 0.35)"},  # Violet
    {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e3a8a", "shadow": "rgba(59, 130, 246, 0.35)"},  # Blue
    {"bg": "#ccfbf1", "border": "#14b8a6", "text": "#0f766e", "shadow": "rgba(20, 184, 166, 0.35)"},  # Teal
    {"bg": "#dcfce7", "border": "#22c55e", "text": "#14532d", "shadow": "rgba(34, 197, 94, 0.35)"},  # Green
    {"bg": "#fef3c7", "border": "#f59e0b", "text": "#78350f", "shadow": "rgba(245, 158, 11, 0.35)"},  # Amber
    {"bg": "#ffedd5", "border": "#f97316", "text": "#7c2d12", "shadow": "rgba(249, 115, 22, 0.35)"},  # Orange
    {"bg": "#fee2e2", "border": "#ef4444", "text": "#7f1d1d", "shadow": "rgba(239, 68, 68, 0.35)"},  # Red
    {"bg": "#fce7f3", "border": "#ec4899", "text": "#831843", "shadow": "rgba(236, 72, 153, 0.35)"},  # Pink
    {"bg": "#f3e8ff", "border": "#a855f7", "text": "#581c87", "shadow": "rgba(168, 85, 247, 0.35)"},  # Purple
    {"bg": "#e0e7ff", "border": "#6366f1", "text": "#312e81", "shadow": "rgba(99, 102, 241, 0.35)"},  # Indigo
    {"bg": "#ecfccb", "border": "#84cc16", "text": "#365314", "shadow": "rgba(132, 204, 22, 0.35)"},  # Lime
    {"bg": "#cffafe", "border": "#06b6d4", "text": "#164e63", "shadow": "rgba(6, 182, 212, 0.35)"},  # Cyan
    {"bg": "#ffe4e6", "border": "#f43f5e", "text": "#881337", "shadow": "rgba(244, 63, 94, 0.35)"},  # Rose
    {"bg": "#fae8ff", "border": "#d946ef", "text": "#701a75", "shadow": "rgba(217, 70, 239, 0.35)"},  # Fuchsia
    {"bg": "#e0f2fe", "border": "#0284c7", "text": "#0c4a6e", "shadow": "rgba(2, 132, 199, 0.35)"},  # Sky
    {"bg": "#fef9c3", "border": "#eab308", "text": "#713f12", "shadow": "rgba(234, 179, 8, 0.35)"},  # Yellow
]

PRESETS = {
    "👋 Hello World & Punctuation": "Hello, world! Welcome to the interactive Tiktokenizer visualizer. How does Byte Pair Encoding work?",
    "🐍 Python Fibonacci Code": """def fibonacci(n: int) -> list[int]:
    \"\"\"Return first n Fibonacci numbers with recursion & memoization.\"\"\"
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]""",
    "🌍 Multilingual & Emojis": "OpenAI GPT-4o supports multilingual tokenization! 🚀✨\nहिंदी: नमस्ते दुनिया, कृत्रिम बुद्धिमत्ता का युग आ गया है!\n日本語: こんにちは世界！AI技術の革新です。\nالعربية: مرحباً بالعالم! الذكاء الاصطناعي يغير المستقبل.\nFrançais: Bonjour le monde! L'intelligence artificielle est magique 🌟",
    "🔢 Numbers, Math & Currencies": "Math equation: 1234567890 + 9876543210 = 11111111100\nConstants: π ≈ 3.1415926535, e ≈ 2.718281828\nCurrency: $49.99 USD = €46.20 EUR = ₹4,180 INR (Discount: 15% off!)",
    "🔒 Special Tokens & ChatML": "<|endoftext|>\n<|im_start|>system\nYou are a helpful and knowledgeable AI assistant.<|im_end|>\n<|im_start|>user\nExplain BPE in one sentence.<|im_end|>\n<|im_start|>assistant\nBPE merges frequent character bytes into subwords.<|im_end|>",
}

# Inject Ultra-Colorful Modern Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

/* Aurora Hero Banner */
.aurora-hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 25%, #ec4899 55%, #f97316 85%, #eab308 100%);
    background-size: 200% 200%;
    animation: gradientShift 10s ease infinite;
    border-radius: 20px;
    padding: 30px 36px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 16px 36px -10px rgba(99, 102, 241, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.aurora-title {
    font-size: 34px;
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -0.8px;
    display: flex;
    align-items: center;
    gap: 12px;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.aurora-sub {
    font-size: 15.5px;
    opacity: 0.96;
    max-width: 820px;
    line-height: 1.6;
    font-weight: 500;
}

.hero-pill-bar {
    display: flex;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.hero-pill {
    background: rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

/* Glowing Metrics Dashboard */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}

.metric-glass-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px 14px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 22px -4px rgba(99, 102, 241, 0.18);
}

.metric-glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}

.card-tokens::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.card-chars::before { background: linear-gradient(90deg, #ec4899, #f43f5e); }
.card-ratio::before { background: linear-gradient(90deg, #10b981, #06b6d4); }
.card-bytes::before { background: linear-gradient(90deg, #f59e0b, #f97316); }
.card-cost::before { background: linear-gradient(90deg, #8b5cf6, #d946ef); }

.m-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #64748b;
    margin-bottom: 4px;
}

.m-val {
    font-size: 26px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a;
    line-height: 1.2;
}

.m-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 3px;
    font-weight: 500;
}

/* Token Stream Visualizer Container */
.tokens-stage {
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px;
    min-height: 140px;
    max-height: 520px;
    overflow-y: auto;
    line-height: 2.3;
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    white-space: pre-wrap;
    word-break: break-word;
    box-shadow: 0 4px 20px -3px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

.tok-chip {
    display: inline-block;
    padding: 3.5px 7.5px;
    margin: 2px 2px;
    border-radius: 7px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14.5px;
    font-weight: 600;
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

.tok-chip:hover {
    transform: translateY(-2px) scale(1.08);
    filter: brightness(0.92);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.16);
    z-index: 50;
    position: relative;
}

.whitespace-glyph {
    opacity: 0.45;
    font-size: 13px;
    user-select: none;
    font-weight: 700;
}

.newline-glyph {
    opacity: 0.75;
    font-size: 13px;
    user-select: none;
    color: #ef4444;
    font-weight: 800;
}

/* Multi-Model Comparison Cards */
.compare-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
}

.compare-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 12px;
}

.comp-model-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 700;
    color: #4f46e5;
}

.comp-pill-badge {
    background: #e0e7ff;
    color: #4338ca;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
}

.comp-stat-badge {
    font-size: 13.5px;
    color: #475569;
}

.gain-tag-better {
    background: #dcfce7;
    color: #15803d;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 12px;
}

.gain-tag-worse {
    background: #fee2e2;
    color: #b91c1c;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# Helper Functions
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
        return '<div style="color: #94a3b8; font-style: italic; padding: 20px; text-align: center;">Enter text above to see tokens rendered.</div>'
    
    chips = []
    for idx, t in enumerate(tokens):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        try:
            b = enc.decode_single_token_bytes(t)
        except Exception:
            b = b"<err>"
        
        display_str, plain_str = format_token_text(b, show_whitespace)
        hex_id = f"0x{t:04X}"
        byte_hex = " ".join(f"0x{x:02X}" for x in b)
        tooltip = f"Token #{idx} | ID: {t} ({hex_id}) | Bytes: {len(b)} ({byte_hex}) | String: {repr(plain_str)}"
        style = f"background-color: {color['bg']}; border: 1.5px solid {color['border']}; color: {color['text']};"
        chips.append(f'<span class="tok-chip" title="{html.escape(tooltip)}" style="{style}">{display_str}</span>')
    
    return f'<div class="tokens-stage">{"".join(chips)}</div>'

# Aurora Gradient Hero Header
st.markdown("""
<div class="aurora-hero">
    <div class="aurora-title">TikTokenizer Studio 🔠✨</div>
    <div class="aurora-sub">
        Explore Byte Pair Encoding (BPE) interactively. Visualize token boundaries with vivid colors, analyze compression efficiency, inspect raw byte sequences, and compare OpenAI models side-by-side.
    </div>
    <div class="hero-pill-bar">
        <span class="hero-pill">⚡ o200k_base (GPT-4o)</span>
        <span class="hero-pill">🚀 cl100k_base (GPT-4)</span>
        <span class="hero-pill">🐍 Codex p50k</span>
        <span class="hero-pill">💎 Multi-Color Chips</span>
        <span class="hero-pill">📊 Real-Time Cost Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Tokenizer Settings")
selected_model = st.sidebar.selectbox(
    "Select BPE Encoding",
    options=list(ENCODING_INFO.keys()),
    index=0,
    format_func=lambda k: ENCODING_INFO[k]["label"],
)

info = ENCODING_INFO[selected_model]
st.sidebar.info(f"**Vocab Size**: `{info['vocab_size']}`\n\n**Models**: {info['models']}\n\n{info['description']}")

st.sidebar.markdown("### 📋 Quick Preset Library")
selected_preset = st.sidebar.selectbox("Choose a prompt example", options=["-- Custom Text --"] + list(PRESETS.keys()))

st.sidebar.markdown("### 👁️ Visual Display Options")
show_ws = st.sidebar.checkbox("Show whitespace symbols (`·` space, `↵` newline, `⇥` tab)", value=True)
allow_special = st.sidebar.checkbox("Allow Special Tokens (e.g. `<|endoftext|>`)", value=True)

# Main Tabs
tab_encode, tab_decode, tab_compare, tab_guide = st.tabs([
    "🔍 Visual Tokenizer (Encode)",
    "🔄 Token Decoder (Decode)",
    "⚖️ Multi-Model Comparison",
    "📚 BPE Educational Deep Dive",
])

# Default text
default_text = "Hello, world! 🌍 Welcome to TikTokenizer Studio.\nTry typing or pasting any text, code, or emojis to visualize the tokens in real-time."
if selected_preset != "-- Custom Text --" and selected_preset in PRESETS:
    default_text = PRESETS[selected_preset]

# TAB 1: ENCODE
with tab_encode:
    user_text = st.text_area(
        "Enter text to tokenize:",
        value=default_text,
        height=140,
        placeholder="Type or paste text, code, or prompts here...",
        help="Type any text to see token boundaries update in real-time"
    )
    
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

    tok_count = len(tokens)
    char_count = len(user_text)
    byte_count = len(user_text.encode("utf-8"))
    ratio = (char_count / tok_count) if tok_count > 0 else 0.0
    cost_per_m = info.get("cost_input_per_m", 2.50)
    est_cost = (tok_count / 1_000_000.0) * cost_per_m

    # Glossy Modern Metric Cards
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-glass-card card-tokens">
            <div class="m-label">Tokens</div>
            <div class="m-val">{tok_count:,}</div>
            <div class="m-sub">Generated chunks</div>
        </div>
        <div class="metric-glass-card card-chars">
            <div class="m-label">Characters</div>
            <div class="m-val">{char_count:,}</div>
            <div class="m-sub">String length</div>
        </div>
        <div class="metric-glass-card card-ratio">
            <div class="m-label">Chars / Token</div>
            <div class="m-val">{ratio:.2f}</div>
            <div class="m-sub">Compression power</div>
        </div>
        <div class="metric-glass-card card-bytes">
            <div class="m-label">Bytes</div>
            <div class="m-val">{byte_count:,}</div>
            <div class="m-sub">UTF-8 payload</div>
        </div>
        <div class="metric-glass-card card-cost">
            <div class="m-label">Est. Cost</div>
            <div class="m-val">${est_cost:.6f}</div>
            <div class="m-sub">@ ${cost_per_m:.2f} / 1M tok</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎨 Color-Coded Token Stream")
    chips_html = render_chips_html(tokens, enc, show_ws)
    st.markdown(chips_html, unsafe_allow_html=True)
    st.caption("💡 **Tip:** Hover your mouse over any token badge to inspect its Decimal ID, Hex code, UTF-8 bytes, and string.")

    with st.expander("📋 Export Token IDs & Sequences", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Python / JSON Array:**")
            st.code(json.dumps(tokens), language="json")
        with c2:
            st.markdown("**Space-Separated Integers:**")
            st.code(" ".join(str(t) for t in tokens), language="text")

    with st.expander("📊 Detailed Token Breakdown Table", expanded=False):
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
                    "Length": f"{len(b)} bytes",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# TAB 2: DECODE
with tab_decode:
    st.markdown("### 🔄 Decode Token IDs Back to Text")
    st.markdown("Paste integer token IDs (e.g. from an API response, logs, or JSON array):")
    
    decode_model = st.selectbox(
        "Decoder Model Encoding",
        options=list(ENCODING_INFO.keys()),
        index=0,
        format_func=lambda k: ENCODING_INFO[k]["label"],
        key="decode_model_key"
    )
    decode_input = st.text_area(
        "Token IDs Input",
        value="13225, 11, 2375, 0, 130321, 235",
        height=90,
        placeholder="e.g. 13225, 11, 2375, 0"
    )
    
    if decode_input.strip():
        nums = re.findall(r"\d+", decode_input)
        if nums:
            dec_tokens = [int(n) for n in nums]
            d_enc = get_cached_encoding(decode_model)
            try:
                decoded_str = d_enc.decode(dec_tokens)
                st.markdown("**Decoded String:**")
                st.text_area("Decoded Output Text", value=decoded_str, height=100, label_visibility="collapsed")
                st.markdown("**Reconstructed Token Pills:**")
                st.markdown(render_chips_html(dec_tokens, d_enc, show_ws), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Decoding failed: {e}")
        else:
            st.warning("No valid integer token IDs found in input.")

# TAB 3: COMPARE
with tab_compare:
    st.markdown("### ⚖️ Multi-Model Side-by-Side Comparison")
    st.markdown("See how BPE tokenizers compress the same prompt across OpenAI model generations:")
    
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
            
            if cnt == base_tokens:
                diff_tag = '<span class="gain-tag-better">Baseline (0%)</span>'
                diff_str = "Baseline (0%)"
            else:
                pct = ((cnt - base_tokens) / base_tokens) * 100
                diff_tag = f'<span class="gain-tag-worse">+{pct:.1f}% more tokens</span>'
                diff_str = f"+{pct:.1f}%"
            
            comp_rows.append({
                "Encoding": m,
                "Target Models": ENCODING_INFO[m]["models"],
                "Vocab Size": ENCODING_INFO[m]["vocab_size"],
                "Tokens": cnt,
                "Chars/Token": round(chars_tok, 2),
                "Efficiency vs o200k": diff_str,
            })
            
            with st.container():
                st.markdown(f"""
                <div class="compare-box">
                    <div class="compare-head">
                        <div>
                            <span class="comp-model-name">{m}</span>
                            <span class="comp-pill-badge">{ENCODING_INFO[m]['models']}</span>
                        </div>
                        <div class="comp-stat-badge">
                            <strong>{cnt}</strong> tokens &bull; {chars_tok:.2f} chars/tok &bull; {diff_tag}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(render_chips_html(m_tokens, m_enc, show_ws), unsafe_allow_html=True)
        
        st.markdown("#### 📊 Summary Benchmark Table")
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

# TAB 4: GUIDE
with tab_guide:
    st.markdown("""
    ## 📚 Understanding Byte Pair Encoding (BPE) & Tiktoken

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
