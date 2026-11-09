import html
import re
import json
import tiktoken
import gradio as gr
import pandas as pd

# Supported Tiktoken Encodings
ENCODING_INFO = {
    "o200k_base": {
        "label": "o200k_base (GPT-4o, GPT-4o mini)",
        "models": "GPT-4o, GPT-4o mini",
        "vocab_size": "200,019",
        "description": "OpenAI's latest state-of-the-art tokenizer. Drastically improves compression for multilingual text (up to 40% fewer tokens), code, and math.",
        "cost_input_per_m": 2.50,
        "cost_output_per_m": 10.00,
    },
    "cl100k_base": {
        "label": "cl100k_base (GPT-4, GPT-3.5 Turbo)",
        "models": "GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-ada-002, text-embedding-3",
        "vocab_size": "100,277",
        "description": "The standard workhorse tokenizer for GPT-4 and GPT-3.5 Turbo with a 100k vocabulary.",
        "cost_input_per_m": 10.00,
        "cost_output_per_m": 30.00,
    },
    "p50k_base": {
        "label": "p50k_base (Codex, text-davinci-002/003)",
        "models": "Codex, text-davinci-002, text-davinci-003, code-davinci-002",
        "vocab_size": "50,281",
        "description": "Tokenizer optimized for code generation and older Davinci models.",
        "cost_input_per_m": 20.00,
        "cost_output_per_m": 20.00,
    },
    "p50k_edit": {
        "label": "p50k_edit (text-davinci-edit-001)",
        "models": "text-davinci-edit-001, code-davinci-edit-001",
        "vocab_size": "50,284",
        "description": "Specialized tokenizer for OpenAI edit and insertion endpoints.",
        "cost_input_per_m": 20.00,
        "cost_output_per_m": 20.00,
    },
    "r50k_base": {
        "label": "r50k_base (GPT-3 legacy)",
        "models": "GPT-3 (davinci, curie, babbage, ada)",
        "vocab_size": "50,257",
        "description": "Original GPT-3 50k vocabulary tokenizer.",
        "cost_input_per_m": 20.00,
        "cost_output_per_m": 20.00,
    },
    "gpt2": {
        "label": "gpt2 (GPT-2)",
        "models": "GPT-2",
        "vocab_size": "50,257",
        "description": "The foundational BPE tokenizer introduced in the 2019 GPT-2 paper.",
        "cost_input_per_m": 0.00,
        "cost_output_per_m": 0.00,
    },
}

# 16 vibrant token colors designed for high contrast and readability
COLOR_PALETTE = [
    {"bg": "#ede9fe", "border": "#8b5cf6", "text": "#4c1d95"},  # Violet
    {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e3a8a"},  # Blue
    {"bg": "#ccfbf1", "border": "#14b8a6", "text": "#0f766e"},  # Teal
    {"bg": "#dcfce7", "border": "#22c55e", "text": "#14532d"},  # Green
    {"bg": "#fef3c7", "border": "#f59e0b", "text": "#78350f"},  # Amber
    {"bg": "#ffedd5", "border": "#f97316", "text": "#7c2d12"},  # Orange
    {"bg": "#fee2e2", "border": "#ef4444", "text": "#7f1d1d"},  # Red
    {"bg": "#fce7f3", "border": "#ec4899", "text": "#831843"},  # Pink
    {"bg": "#f3e8ff", "border": "#a855f7", "text": "#581c87"},  # Purple
    {"bg": "#e0e7ff", "border": "#6366f1", "text": "#312e81"},  # Indigo
    {"bg": "#ecfccb", "border": "#84cc16", "text": "#365314"},  # Lime
    {"bg": "#cffafe", "border": "#06b6d4", "text": "#164e63"},  # Cyan
    {"bg": "#ffe4e6", "border": "#f43f5e", "text": "#881337"},  # Rose
    {"bg": "#fae8ff", "border": "#d946ef", "text": "#701a75"},  # Fuchsia
    {"bg": "#e0f2fe", "border": "#0284c7", "text": "#0c4a6e"},  # Sky
    {"bg": "#fef9c3", "border": "#eab308", "text": "#713f12"},  # Yellow
]

# Preset Samples
EXAMPLES = {
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
    "🔢 Numbers, Math & Symbols": "Calculate: 1234567890 + 9876543210 = 11111111100\nPi: 3.141592653589793\nPrice: $49.99 (Discount: 15% off) -> Total: $42.49",
    "🔒 Special Tokens & Edge Cases": "<|endoftext|>\n<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>",
}

def get_encoding_instance(encoding_name: str):
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return tiktoken.get_encoding("o200k_base")

def format_token_text_visual(raw_bytes: bytes, show_whitespace: bool) -> tuple[str, str]:
    """Returns (display_html_text, plain_decoded_str)"""
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
        display = "".join(out)
    else:
        # Preserve newlines in display
        display = html.escape(decoded).replace("\n", "<br/>")

    return display, plain

def render_tokens_html(tokens: list[int], enc, show_whitespace: bool, container_id: str = "main-tokens") -> str:
    if not tokens:
        return '<div class="empty-state">Enter text above to see tokens rendered in real-time.</div>'

    chips_html = []
    for idx, token_id in enumerate(tokens):
        color_item = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        try:
            raw_bytes = enc.decode_single_token_bytes(token_id)
        except Exception:
            raw_bytes = b"<err>"
        
        display_text, plain_text = format_token_text_visual(raw_bytes, show_whitespace)
        
        hex_id = f"0x{token_id:04X}"
        byte_repr = repr(raw_bytes)
        escaped_plain = repr(plain_text)
        tooltip = f"Token #{idx} | ID: {token_id} ({hex_id}) | Bytes: {len(raw_bytes)} | String: {escaped_plain}"
        
        style = (
            f"background-color: {color_item['bg']}; "
            f"border: 1px solid {color_item['border']}; "
            f"color: {color_item['text']};"
        )
        
        chip = (
            f'<span class="tok-chip" '
            f'data-idx="{idx}" '
            f'data-id="{token_id}" '
            f'data-hex="{hex_id}" '
            f'data-bytes="{html.escape(byte_repr)}" '
            f'data-text="{html.escape(escaped_plain)}" '
            f'title="{html.escape(tooltip)}" '
            f'style="{style}">'
            f'{display_text}'
            f'</span>'
        )
        chips_html.append(chip)

    chips_str = "".join(chips_html)
    
    # Render with real-time inspection bar
    return f"""
    <div class="tokens-wrapper">
        <div class="token-container" id="{container_id}">
            {chips_str}
        </div>
        <div class="token-inspector-strip" id="strip-{container_id}">
            <span class="inspector-prompt">💡 <strong>Hover over any token</strong> to inspect its Token ID, UTF-8 byte breakdown, and hex representation.</span>
        </div>
    </div>
    <script>
    (function() {{
        const container = document.getElementById("{container_id}");
        const strip = document.getElementById("strip-{container_id}");
        if (!container || !strip) return;
        
        const chips = container.querySelectorAll(".tok-chip");
        chips.forEach(chip => {{
            chip.addEventListener("mouseenter", () => {{
                strip.innerHTML = `
                    <div class="inspector-active">
                        <span class="badge-idx">Token #${{chip.dataset.idx}}</span>
                        <span class="badge-id">ID: <strong>${{chip.dataset.id}}</strong> (<code>${{chip.dataset.hex}}</code>)</span>
                        <span class="badge-bytes">Bytes: <code>${{chip.dataset.bytes}}</code></span>
                        <span class="badge-str">String: <code>${{chip.dataset.text}}</code></span>
                    </div>
                `;
            }});
        }});
    }})();
    </script>
    """

def build_metrics_html(tokens_count: int, char_count: int, byte_count: int, encoding_name: str) -> str:
    ratio = (char_count / tokens_count) if tokens_count > 0 else 0.0
    info = ENCODING_INFO.get(encoding_name, {})
    cost_per_m = info.get("cost_input_per_m", 2.50)
    est_cost = (tokens_count / 1_000_000.0) * cost_per_m

    return f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Tokens</div>
            <div class="metric-value">{tokens_count:,}</div>
            <div class="metric-sub">Total tokens generated</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Characters</div>
            <div class="metric-value">{char_count:,}</div>
            <div class="metric-sub">Raw text length</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Chars / Token</div>
            <div class="metric-value">{ratio:.2f}</div>
            <div class="metric-sub">Higher = better compression</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Bytes</div>
            <div class="metric-value">{byte_count:,}</div>
            <div class="metric-sub">UTF-8 encoded size</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Est. Cost (1M tokens)</div>
            <div class="metric-value">${est_cost:.6f}</div>
            <div class="metric-sub">Based on ${cost_per_m:.2f}/M tokens</div>
        </div>
    </div>
    """

def tokenize_pipeline(text: str, encoding_name: str, show_whitespace: bool, allow_special: bool):
    if not text:
        return (
            '<div class="empty-state">Enter text above to inspect tokens.</div>',
            build_metrics_html(0, 0, 0, encoding_name),
            "[]",
            "",
            pd.DataFrame(columns=["Index", "Token ID", "Hex", "String", "Bytes", "Length"]),
            f"**Tokenizer**: `{encoding_name}` | **Vocab Size**: {ENCODING_INFO.get(encoding_name, {}).get('vocab_size', 'N/A')}"
        )

    enc = get_encoding_instance(encoding_name)
    
    try:
        if allow_special:
            tokens = enc.encode(text, allowed_special="all")
        else:
            tokens = enc.encode(text, disallowed_special=())
    except Exception as e:
        err_msg = f'<div class="error-box">Tokenization error: {html.escape(str(e))}</div>'
        return err_msg, "", "[]", "", pd.DataFrame(), str(e)

    raw_bytes_list = []
    table_rows = []
    total_bytes = 0

    for idx, t in enumerate(tokens):
        try:
            b = enc.decode_single_token_bytes(t)
        except Exception:
            b = b"<err>"
        raw_bytes_list.append(b)
        total_bytes += len(b)
        
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            s = b.decode("utf-8", errors="replace")
            
        table_rows.append({
            "Index": f"#{idx}",
            "Token ID": t,
            "Hex": f"0x{t:04X}",
            "String": repr(s)[1:-1],
            "Bytes": repr(b),
            "Length": len(b)
        })

    html_tokens = render_tokens_html(tokens, enc, show_whitespace, "main-encode-tokens")
    metrics_html = build_metrics_html(len(tokens), len(text), total_bytes, encoding_name)
    token_ids_json = json.dumps(tokens)
    token_ids_space = " ".join(str(t) for t in tokens)
    df_table = pd.DataFrame(table_rows)
    
    info = ENCODING_INFO.get(encoding_name, {})
    status_md = f"**Tokenizer**: `{encoding_name}` | **Models**: {info.get('models', '')} | **Vocab**: {info.get('vocab_size', '')}"

    return html_tokens, metrics_html, token_ids_json, token_ids_space, df_table, status_md

def decode_pipeline(token_ids_str: str, encoding_name: str, show_whitespace: bool):
    if not token_ids_str.strip():
        return "", '<div class="empty-state">Enter token IDs above to decode.</div>', 0, 0

    enc = get_encoding_instance(encoding_name)
    
    found_nums = re.findall(r"\d+", token_ids_str)
    if not found_nums:
        return "No valid integers found in input.", '<div class="error-box">Invalid token ID format. Please provide integers.</div>', 0, 0
    
    tokens = [int(num) for num in found_nums]
    
    try:
        decoded_text = enc.decode(tokens)
    except Exception as e:
        return f"Decoding error: {e}", f'<div class="error-box">{html.escape(str(e))}</div>', len(tokens), 0

    html_tokens = render_tokens_html(tokens, enc, show_whitespace, "main-decode-tokens")
    return decoded_text, html_tokens, len(tokens), len(decoded_text)

def multi_model_compare(text: str, show_whitespace: bool):
    if not text.strip():
        return pd.DataFrame(), '<div class="empty-state">Enter text to compare tokenizers side-by-side.</div>'

    models = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base", "gpt2"]
    rows = []
    comparisons_html = []

    base_tokens_count = None

    for idx, m in enumerate(models):
        enc = get_encoding_instance(m)
        try:
            tokens = enc.encode(text, allowed_special="all")
        except Exception:
            tokens = enc.encode(text, disallowed_special=())
        
        t_count = len(tokens)
        char_count = len(text)
        ratio = (char_count / t_count) if t_count > 0 else 0.0
        info = ENCODING_INFO.get(m, {})

        if base_tokens_count is None:
            base_tokens_count = t_count
            diff_pct = "Baseline (0%)"
        else:
            diff = ((t_count - base_tokens_count) / base_tokens_count) * 100
            diff_pct = f"+{diff:.1f}% more tokens" if diff > 0 else f"{diff:.1f}% tokens"

        rows.append({
            "Encoding": m,
            "Target Models": info.get("models", ""),
            "Vocab Size": info.get("vocab_size", ""),
            "Tokens": t_count,
            "Chars/Token": round(ratio, 2),
            "Efficiency vs o200k": diff_pct
        })

        preview_html = render_tokens_html(tokens, enc, show_whitespace, f"compare-tokens-{idx}")
        comparisons_html.append(f"""
        <div class="model-compare-block">
            <div class="model-compare-header">
                <div>
                    <span class="compare-title">{m}</span>
                    <span class="compare-badge">{info.get('models', '')}</span>
                </div>
                <div class="compare-stat">
                    <strong>{t_count:,}</strong> tokens &bull; {ratio:.2f} chars/tok &bull; <span class="diff-tag">{diff_pct}</span>
                </div>
            </div>
            {preview_html}
        </div>
        """)

    df_comp = pd.DataFrame(rows)
    combined_html = "".join(comparisons_html)
    return df_comp, combined_html

# Styling & Layout
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.hero-banner {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 25%, #ec4899 55%, #f97316 85%, #eab308 100%);
    background-size: 200% 200%;
    animation: gradientShift 10s ease infinite;
    border-radius: 18px;
    padding: 26px 32px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 14px 34px -8px rgba(99, 102, 241, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
}

.hero-subtitle {
    font-size: 15px;
    opacity: 0.96;
    margin: 0;
    line-height: 1.6;
}

.tokens-wrapper {
    margin-top: 8px;
    margin-bottom: 8px;
}

.token-container {
    padding: 16px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px 12px 0 0;
    min-height: 120px;
    max-height: 480px;
    overflow-y: auto;
    line-height: 2.2;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14.5px;
    white-space: pre-wrap;
    word-break: break-word;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02);
}

.dark .token-container {
    background: #0f172a;
    border-color: #334155;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

.token-inspector-strip {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 8px 14px;
    font-size: 12.5px;
    color: #475569;
    min-height: 38px;
    display: flex;
    align-items: center;
}

.dark .token-inspector-strip {
    background: #1e293b;
    border-color: #334155;
    color: #94a3b8;
}

.inspector-prompt {
    color: #64748b;
    font-size: 12px;
}

.inspector-active {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
}

.badge-idx {
    background: #3b82f6;
    color: white;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11.5px;
}

.badge-id, .badge-bytes, .badge-str {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

.tok-chip {
    display: inline-block;
    padding: 3px 7px;
    margin: 2px 2px;
    border-radius: 6px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14.5px;
    font-weight: 600;
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.tok-chip:hover {
    filter: brightness(0.92);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18);
    transform: translateY(-2px) scale(1.08);
    z-index: 50;
    position: relative;
}

.whitespace-glyph {
    opacity: 0.45;
    font-size: 13px;
    user-select: none;
}

.newline-glyph {
    opacity: 0.65;
    font-size: 13px;
    user-select: none;
    color: #ef4444;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-top: 12px;
    margin-bottom: 14px;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(99, 102, 241, 0.15);
}

.dark .metric-card {
    background: #1e293b;
    border-color: #334155;
}

.metric-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #64748b;
    margin-bottom: 4px;
}

.dark .metric-label {
    color: #94a3b8;
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
}

.dark .metric-value {
    color: #f8fafc;
}

.metric-sub {
    font-size: 10.5px;
    color: #94a3b8;
    margin-top: 3px;
}

.empty-state {
    padding: 30px;
    text-align: center;
    color: #94a3b8;
    font-style: italic;
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
}

.dark .empty-state {
    background: #1e293b;
    border-color: #334155;
    color: #64748b;
}

.model-compare-block {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.dark .model-compare-block {
    background: #0f172a;
    border-color: #334155;
}

.model-compare-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #f1f5f9;
}

.dark .model-compare-header {
    border-bottom-color: #1e293b;
}

.compare-title {
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    color: #4f46e5;
    margin-right: 8px;
}

.compare-badge {
    background: #eef2ff;
    color: #4338ca;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

.dark .compare-badge {
    background: #312e81;
    color: #c7d2fe;
}

.compare-stat {
    font-size: 13px;
    color: #64748b;
}

.dark .compare-stat {
    color: #94a3b8;
}

.diff-tag {
    font-weight: 600;
    color: #0369a1;
}

.dark .diff-tag {
    color: #38bdf8;
}
"""

def create_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="TikTokenizer Studio 🔠") as demo:
        gr.HTML("""
        <div class="hero-banner">
            <h1 class="hero-title">TikTokenizer Studio 🔠</h1>
            <p class="hero-subtitle">Interactive Byte Pair Encoding (BPE) Visualizer & Multi-Model Token Inspector for OpenAI & LLM Tokenizers</p>
        </div>
        """)

        with gr.Tabs():
            # Tab 1: Visual Tokenizer (Encode)
            with gr.TabItem("🔍 Visual Tokenizer (Encode)", id="tab_encode"):
                with gr.Row():
                    with gr.Column(scale=8):
                        encoding_dropdown = gr.Dropdown(
                            choices=list(ENCODING_INFO.keys()),
                            value="o200k_base",
                            label="Tokenizer Encoding",
                            info="Select BPE encoding algorithm"
                        )
                    with gr.Column(scale=4):
                        example_selector = gr.Dropdown(
                            choices=list(EXAMPLES.keys()),
                            label="Load Example Snippet",
                            value=None
                        )

                with gr.Row():
                    show_ws_cb = gr.Checkbox(
                        label="Show whitespace glyphs (· space, ↵ newline, ⇥ tab)",
                        value=True
                    )
                    allow_spec_cb = gr.Checkbox(
                        label="Allow Special Tokens (e.g. <|endoftext|>)",
                        value=True
                    )

                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Type or paste text here to see how it splits into tokens...",
                    lines=5,
                    value="Hello, world! 🌍 Welcome to TikTokenizer Studio.\nTry typing or pasting any text, code, or emojis to visualize the tokens in real-time."
                )

                status_label = gr.Markdown(
                    value=f"**Tokenizer**: `o200k_base` | **Models**: {ENCODING_INFO['o200k_base']['models']} | **Vocab**: {ENCODING_INFO['o200k_base']['vocab_size']}"
                )

                metrics_output = gr.HTML(label="Metrics Dashboard")
                tokens_html_output = gr.HTML(label="Color-Coded Token Stream")

                with gr.Accordion("📋 Token IDs & Sequence Export", open=False):
                    with gr.Row():
                        token_json_output = gr.Textbox(
                            label="JSON / Python List [ID, ID, ...]",
                            show_copy_button=True,
                            lines=2
                        )
                        token_space_output = gr.Textbox(
                            label="Space-Separated IDs",
                            show_copy_button=True,
                            lines=2
                        )

                with gr.Accordion("📊 Detailed Token Inspector Table", open=False):
                    tokens_table = gr.DataFrame(
                        headers=["Index", "Token ID", "Hex", "String", "Bytes", "Length"],
                        datatype=["str", "number", "str", "str", "str", "number"],
                        interactive=False
                    )

                input_components = [input_text, encoding_dropdown, show_ws_cb, allow_spec_cb]
                output_components = [
                    tokens_html_output,
                    metrics_output,
                    token_json_output,
                    token_space_output,
                    tokens_table,
                    status_label
                ]

                input_text.change(fn=tokenize_pipeline, inputs=input_components, outputs=output_components)
                encoding_dropdown.change(fn=tokenize_pipeline, inputs=input_components, outputs=output_components)
                show_ws_cb.change(fn=tokenize_pipeline, inputs=input_components, outputs=output_components)
                allow_spec_cb.change(fn=tokenize_pipeline, inputs=input_components, outputs=output_components)

                def on_example_selected(key):
                    if key and key in EXAMPLES:
                        return EXAMPLES[key]
                    return gr.update()

                example_selector.change(fn=on_example_selected, inputs=[example_selector], outputs=[input_text])

            # Tab 2: Token Decoder (Decode)
            with gr.TabItem("🔄 Token Decoder (Decode)", id="tab_decode"):
                gr.Markdown("### Decode Token IDs back to Original Text")
                gr.Markdown("Enter a list of integer token IDs (comma-separated, space-separated, or JSON array format like `[13225, 11, 2375, 0]`).")
                
                with gr.Row():
                    decode_encoding = gr.Dropdown(
                        choices=list(ENCODING_INFO.keys()),
                        value="o200k_base",
                        label="Decoder Encoding"
                    )
                    decode_ws_cb = gr.Checkbox(
                        label="Show whitespace glyphs",
                        value=True
                    )

                decode_input = gr.Textbox(
                    label="Token IDs Input",
                    placeholder="e.g. 13225, 11, 2375, 0, 130321, 235",
                    value="13225, 11, 2375, 0, 130321, 235",
                    lines=3
                )
                
                decode_btn = gr.Button("Decode Tokens 🚀", variant="primary")

                with gr.Row():
                    with gr.Column():
                        decoded_text_out = gr.Textbox(
                            label="Decoded Plain Text",
                            lines=5,
                            show_copy_button=True
                        )
                    with gr.Column():
                        with gr.Row():
                            dec_tok_count = gr.Number(label="Total Tokens Decoded", value=6)
                            dec_char_count = gr.Number(label="Total Characters", value=15)

                decode_html_out = gr.HTML(label="Reconstructed Token Visualizer")

                decode_inputs = [decode_input, decode_encoding, decode_ws_cb]
                decode_outputs = [decoded_text_out, decode_html_out, dec_tok_count, dec_char_count]

                decode_btn.click(fn=decode_pipeline, inputs=decode_inputs, outputs=decode_outputs)
                decode_input.change(fn=decode_pipeline, inputs=decode_inputs, outputs=decode_outputs)

            # Tab 3: Multi-Model Comparison
            with gr.TabItem("⚖️ Multi-Model Comparison", id="tab_compare"):
                gr.Markdown("### Compare BPE Tokenizers Side-by-Side")
                gr.Markdown("See how different generations of tokenizers (from `gpt2` up to `o200k_base`) handle the exact same text, code, or multilingual input.")

                compare_text = gr.Textbox(
                    label="Comparison Text",
                    placeholder="Enter text to compare across GPT-4o, GPT-4, Codex, and GPT-2 tokenizers...",
                    lines=4,
                    value="Artificial Intelligence is revolutionizing natural language processing across multiple languages: हिंदी, 日本語, and Español! 🚀"
                )

                compare_ws_cb = gr.Checkbox(label="Show whitespace glyphs", value=True)
                compare_btn = gr.Button("Run Comparison ⚡", variant="primary")

                gr.Markdown("#### Comparison Summary Table")
                compare_table = gr.DataFrame(
                    headers=["Encoding", "Target Models", "Vocab Size", "Tokens", "Chars/Token", "Efficiency vs o200k"],
                    datatype=["str", "str", "str", "number", "number", "str"]
                )

                gr.Markdown("#### Visual Tokenizer Previews")
                compare_html = gr.HTML()

                compare_btn.click(
                    fn=multi_model_compare,
                    inputs=[compare_text, compare_ws_cb],
                    outputs=[compare_table, compare_html]
                )
                compare_text.change(
                    fn=multi_model_compare,
                    inputs=[compare_text, compare_ws_cb],
                    outputs=[compare_table, compare_html]
                )

            # Tab 4: Interactive BPE Guide & FAQ
            with gr.TabItem("📚 BPE Deep Dive & Guide", id="tab_guide"):
                gr.Markdown("""
                ## Understanding Byte Pair Encoding (BPE) & Tiktoken

                ### 1. What is Byte Pair Encoding (BPE)?
                Language models don't read words or letters directly—they operate on **Tokens**. 
                BPE is a subword tokenization algorithm that iteratively merges the most frequently occurring byte pairs in a training corpus into single tokens.
                
                ### 2. Why does leading whitespace matter?
                In modern tokenizers like `cl100k_base` and `o200k_base`, spaces are usually fused to the *beginning* of words:
                - `" hello"` (with space) is typically a single token.
                - `"hello"` (without space) is a distinct token with a different Token ID!
                This is why prompt indentation, trailing spaces, or missing spaces can change the tokens sent to the LLM.

                ### 3. The Evolution of OpenAI Tokenizers
                | Tokenizer | Models | Vocab Size | Key Features |
                | :--- | :--- | :--- | :--- |
                | `gpt2` / `r50k_base` | GPT-2, GPT-3 | 50,257 | Baseline BPE vocabulary. Poor compression on non-English text. |
                | `p50k_base` | Codex, text-davinci-002/003 | 50,281 | Tuned for programming languages and source code. |
                | `cl100k_base` | GPT-4, GPT-3.5 Turbo | 100,277 | 2x vocabulary. Substantially better code and foreign language compression. |
                | `o200k_base` | GPT-4o, GPT-4o mini | 200,019 | 200k vocabulary. Up to **40% token reduction** on languages like Hindi, Japanese, Arabic, and Python code! |

                ### 4. Why do Emojis and Non-Latin Scripts use more tokens?
                Unicode characters are encoded in UTF-8:
                - ASCII English: 1 byte per character.
                - Accented Latin: 2 bytes per character.
                - Devanagari (Hindi), Chinese, Arabic: 3 bytes per character.
                - Emojis (e.g. 🌍): 4 bytes (or more for combinations with skin tones and zero-width joiners).
                When a byte sequence hasn't been merged into a single vocabulary entry, BPE must split it into multiple byte tokens. With `o200k_base`, many common multilingual words and emojis have direct tokens!
                """)

        demo.load(
            fn=tokenize_pipeline,
            inputs=[input_text, encoding_dropdown, show_ws_cb, allow_spec_cb],
            outputs=[
                tokens_html_output,
                metrics_output,
                token_json_output,
                token_space_output,
                tokens_table,
                status_label
            ]
        )

    return demo

demo = create_ui()

if __name__ == "__main__":
    demo.launch()
