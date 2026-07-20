"""Gradio演示服务：铁路调度ASR后处理纠错系统.

功能：
1. 支持文本输入和音频上传两种ASR输入方式
2. 可编辑热词（影响ASR识别和后处理术语检索）
3. 分步展示Layer 1-4改写结果
4. 可切换Layer 4模式（baseline/rag/harness）和Prompt版本（v1/v2）
5. 展示当前使用的System Prompt和User Prompt

启动：
    python scripts/gradio_demo.py

默认访问：http://localhost:7860
"""

import base64
import json
import os
import sys
import tempfile
import time
import warnings
import wave
from pathlib import Path

import gradio as gr
import gradio_client.utils as _gradio_client_utils
import requests
from dotenv import load_dotenv

# 抑制Gradio/Starlette版本兼容性产生的弃用警告，保持终端干净
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")

# 加载项目根目录下的 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 避免本机代理设置拦截 Gradio 对 localhost 的内部健康检查请求
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
if os.environ.get("NO_PROXY"):
    os.environ["NO_PROXY"] += ",localhost,127.0.0.1"
else:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 修复当前Gradio/gradio_client版本的schema解析bug：
# additionalProperties为bool(true/false)时，原实现会尝试"const" in schema导致TypeError
_orig_json_schema_to_python_type = _gradio_client_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

from openai import OpenAI

from src.audio_preprocessor import available_methods, denoise_file
from src.pipeline import PostCorrectionPipeline
from src.semantic_refiner import PromptLoader


# ASR来源配置
# 本地部署ASR（VibeVoice格式，base_url不要带/v1后缀）
LOCAL_ASR = {
    "name": "本地ASR(VibeVoice)",
    "base_url": os.getenv("LOCAL_ASR_BASE_URL", "http://192.168.1.119:8015"),
    "model": os.getenv("LOCAL_ASR_MODEL", "vibevoice"),
    "api_key": os.getenv("LOCAL_ASR_API_KEY", "dummy-key-for-local"),
}

# qwen3-asr (vLLM 部署，OpenAI /v1/audio/transcriptions 兼容接口)
QWEN3_ASR = {
    "name": "qwen3-asr (vLLM)",
    "base_url": os.getenv("QWEN3_ASR_BASE_URL", "http://192.168.1.119:8014"),
    "model": os.getenv("QWEN3_ASR_MODEL", "/models/Qwen3-ASR-1.7B"),
    "api_key": os.getenv("QWEN3_ASR_API_KEY", "dummy-key-for-local"),
}

# ElevenLabs ASR（你说成"云知声"的服务）
ELEVENLABS_ASR = {
    "name": "ElevenLabs ASR",
    "base_url": "https://api.elevenlabs.io",
    "model": os.getenv("ELEVENLABS_ASR_MODEL", "scribe_v1"),
    "api_key": os.getenv("ELEVENLABS_ASR_API_KEY", ""),
}

ASR_SOURCES = {
    "本地ASR(VibeVoice)": LOCAL_ASR,
    "qwen3-asr (vLLM)": QWEN3_ASR,
    "ElevenLabs ASR": ELEVENLABS_ASR,
}


# 根据当前环境构建可用的音频增强选项
_AUDIO_ENHANCE_OPTIONS = {"不处理": "none"}
for _m in available_methods():
    if _m == "noisereduce":
        _AUDIO_ENHANCE_OPTIONS["noisereduce (轻量谱减降噪)"] = _m
    elif _m == "deepfilternet":
        _AUDIO_ENHANCE_OPTIONS["deepfilternet (深度学习语音增强，att50)"] = _m
    elif _m == "ffmpeg_afftdn":
        _AUDIO_ENHANCE_OPTIONS["ffmpeg afftdn (系统 ffmpeg 滤波)"] = _m

_DEFAULT_AUDIO_ENHANCE = (
    "deepfilternet (深度学习语音增强，att50)"
    if "deepfilternet (深度学习语音增强，att50)" in _AUDIO_ENHANCE_OPTIONS
    else "不处理"
)


def save_hotwords(hotwords_text: str) -> None:
    """将用户编辑的热词保存到hotwords.json（临时覆盖）."""
    if not hotwords_text.strip():
        return

    hotwords = [w.strip() for w in hotwords_text.replace("，", ",").replace("\n", ",").split(",") if w.strip()]

    lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
    hotwords_file = lexicon_dir / "hotwords.json"

    # 备份原文件
    backup_file = lexicon_dir / "hotwords.json.bak"
    if hotwords_file.exists() and not backup_file.exists():
        hotwords_file.rename(backup_file)

    with open(hotwords_file, "w", encoding="utf-8") as f:
        json.dump(hotwords, f, ensure_ascii=False, indent=2)


def restore_hotwords() -> None:
    """恢复原始热词文件."""
    lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
    hotwords_file = lexicon_dir / "hotwords.json"
    backup_file = lexicon_dir / "hotwords.json.bak"

    if backup_file.exists():
        if hotwords_file.exists():
            hotwords_file.unlink()
        backup_file.rename(hotwords_file)


def _get_audio_duration(path: str) -> float:
    """获取wav音频时长（秒）."""
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def _extract_transcript(raw_output: str) -> str:
    """从VibeVoice ASR的JSON输出中提取纯文本.

    输入形如: [{"Start":0,"End":7.76,"Speaker":0,"Content":"按照列车..."}]
    """
    try:
        items = json.loads(raw_output)
        if isinstance(items, list):
            contents = [
                item["Content"]
                for item in items
                if isinstance(item, dict) and item.get("Content") and item["Content"] != "[Silence]"
            ]
            return " ".join(contents)
    except Exception:
        pass
    return raw_output


def _transcribe_vibevoice(
    audio_path: str,
    base_url: str,
    hotwords_text: str,
    timeout: int = 300,
) -> str:
    """调用VibeVoice ASR服务（本地部署，项目内已验证）."""
    duration = _get_audio_duration(audio_path)
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:audio/wav;base64,{audio_b64}"

    prompt_text = (
        f"This is a {duration:.2f} seconds audio, "
        "please transcribe it with these keys: Start time, End time, Speaker ID, Content"
    )
    # VibeVoice-ASR 通过 prompt 参数提供上下文/热词
    if hotwords_text.strip():
        hw_list = [w.strip() for w in hotwords_text.replace("，", ",").split(",") if w.strip()][:100]
        prompt_text += f"\n\nContext keywords: {', '.join(hw_list)}"

    payload = {
        "model": "vibevoice",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "max_tokens": 32768,
        "temperature": 0.0,
        "stream": True,
        "top_p": 1.0,
    }

    response = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=timeout
    )
    if response.status_code != 200:
        raise RuntimeError(f"ASR请求失败: {response.status_code} {response.text[:500]}")

    full_text = ""
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        json_str = decoded[6:]
        if json_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(json_str)
            content = data["choices"][0]["delta"].get("content", "")
            if content:
                if full_text and content.startswith(full_text):
                    full_text = content
                else:
                    full_text += content
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    return _extract_transcript(full_text)


def _transcribe_qwen3(
    audio_path: str,
    base_url: str,
    model: str,
    hotwords_text: str,
    timeout: int = 120,
) -> str:
    """调用 qwen3-asr (vLLM /v1/audio/transcriptions) 服务.

    热词通过 OpenAI 兼容的 prompt 参数传入，模型会在解码时给予这些词更高权重。
    """
    url = f"{base_url}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer dummy-key-for-local"}

    data = {
        "model": model,
        "language": "zh",
        "response_format": "json",
        "temperature": 0.0,
    }
    # 热词拼接成 prompt，放在转写指令后
    if hotwords_text.strip():
        hw_list = [w.strip() for w in hotwords_text.replace("，", ",").split(",") if w.strip()][:100]
        data["prompt"] = "请准确转写以下铁路调度相关音频。注意识别这些术语：" + "、".join(hw_list)

    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(f"qwen3-asr 请求失败: {response.status_code} {response.text[:800]}")

    result = response.json()
    return result.get("text", "")


def _transcribe_elevenlabs(
    audio_path: str,
    api_key: str,
    model: str,
    hotwords_text: str,
    timeout: int = 120,
) -> str:
    """调用ElevenLabs ASR服务 (/v1/speech-to-text).

    注意：ElevenLabs STT API 目前没有官方热词参数，
    热词文本仅记录到日志，不随ASR请求发送（但会用于后处理纠错）。
    """
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": api_key}

    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {
            "model_id": model,
            "language_code": "zh",
            "tag_audio_events": False,
            "timestamps_granularity": "none",
        }
        response = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs ASR请求失败: {response.status_code} {response.text[:500]}")

    result = response.json()
    return result.get("text", "")


def transcribe_audio(
    audio_path: str,
    asr_source: str,
    asr_base_url: str,
    asr_model: str,
    asr_api_key: str,
    hotwords_text: str,
    audio_enhance: str = "none",
) -> str:
    """调用ASR服务转写音频.

    支持两种来源：
    - 本地ASR(VibeVoice)：项目内已验证的实现，base64+chat/completions，支持热词
    - 云之声ASR(OpenAI兼容)：按OpenAI audio/transcriptions接口调用，热词作为prompt传入

    Args:
        audio_enhance: 音频增强方案，"none" 表示不处理。
    """
    if not audio_path:
        return ""

    # 如需降噪，先处理音频
    enhanced_path = audio_path
    temp_path = None
    if audio_enhance and audio_enhance != "none":
        suffix = f"_enhanced_{audio_enhance}.wav"
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=Path(audio_path).parent)
        os.close(temp_fd)
        try:
            denoise_file(audio_path, temp_path, method=audio_enhance)
            enhanced_path = temp_path
        except Exception as e:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink(missing_ok=True)
            return f"[音频增强失败: {e}]"

    try:
        if "VibeVoice" in asr_source:
            result = _transcribe_vibevoice(enhanced_path, asr_base_url, hotwords_text)
        elif "qwen3" in asr_source.lower():
            result = _transcribe_qwen3(enhanced_path, asr_base_url, asr_model, hotwords_text)
        elif "ElevenLabs" in asr_source:
            result = _transcribe_elevenlabs(enhanced_path, asr_api_key, asr_model, hotwords_text)
        else:
            result = f"[未知ASR来源: {asr_source}]"
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)

    return result


def get_prompts(version: str) -> tuple[str, str]:
    """获取指定版本的system prompt和user template."""
    os.environ["LLM_PROMPT_VERSION"] = version
    system_prompt, user_template = PromptLoader.load()
    return system_prompt, user_template


def _escape_html(s: str) -> str:
    """转义HTML特殊字符并保留换行显示."""
    import html
    return html.escape(s).replace("\n", "<br>")


def diff_to_html(prev_text: str, curr_text: str, title: str = "") -> str:
    """用HTML高亮显示两次文本之间的差异.

    绿色高亮：新增/修改后的内容
    红色删除线：被删除/替换掉的内容
    """
    header = f'<div style="font-weight:bold;color:#333;margin-bottom:4px;">{title}</div>' if title else ""
    if prev_text == curr_text:
        return f'{header}<div style="padding:8px;line-height:1.6;background:#f8f9fa;border-radius:4px;">{_escape_html(curr_text)}</div>'

    import difflib
    sm = difflib.SequenceMatcher(None, prev_text, curr_text)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(_escape_html(curr_text[j1:j2]))
        elif tag == "replace":
            parts.append(f'<span style="background:#ff9999;text-decoration:line-through;">{_escape_html(prev_text[i1:i2])}</span>')
            parts.append(f'<span style="background:#99ff99;font-weight:bold;">{_escape_html(curr_text[j1:j2])}</span>')
        elif tag == "delete":
            parts.append(f'<span style="background:#ff9999;text-decoration:line-through;">{_escape_html(prev_text[i1:i2])}</span>')
        elif tag == "insert":
            parts.append(f'<span style="background:#99ff99;font-weight:bold;">{_escape_html(curr_text[j1:j2])}</span>')

    return f'{header}<div style="padding:8px;line-height:1.6;background:#f8f9fa;border-radius:4px;">{"".join(parts)}</div>'


def correct_diff_html(final_text: str, correct_text: str) -> str:
    """生成最终输出与正确答案的对比HTML."""
    if not correct_text:
        return '<div style="padding:8px;color:#999;">未提供正确答案</div>'
    return diff_to_html(final_text, correct_text, title="正确答案对比（绿色=正确，红色=差异）")


def process_correction(
    input_mode: str,
    text_input: str,
    audio_input: str,
    asr_source: str,
    asr_base_url: str,
    asr_model: str,
    asr_api_key: str,
    hotwords_text: str,
    layers: list[str],
    semantic_mode: str,
    prompt_version: str,
    correct_text: str,
    audio_enhance: str = "none",
):
    """执行纠错并流式返回各层结果和Prompt.

    每完成一步yield一次，让前端实时展示中间结果。
    """

    # 把界面显示名称映射为内部方法 key
    audio_enhance = _AUDIO_ENHANCE_OPTIONS.get(audio_enhance, "none")

    # 1. 获取ASR原始文本
    # 初始状态
    yield "", "", "", "", "", "", "", "", "", "", "Step 0: ASR转写中..."

    if input_mode == "音频上传":
        original_text = transcribe_audio(
            audio_input,
            asr_source,
            asr_base_url,
            asr_model,
            asr_api_key,
            hotwords_text,
            audio_enhance=audio_enhance,
        )
    else:
        original_text = text_input

    if not original_text or original_text.startswith(("[ASR调用失败", "[音频增强失败", "[未知ASR来源")):
        yield original_text, "", "", "", "", "", "", "", "", "", "ASR调用失败或输入为空"
        return

    # 2. 保存热词并重新初始化pipeline（使热词生效）
    save_hotwords(hotwords_text)

    # 3. 设置prompt版本
    os.environ["LLM_PROMPT_VERSION"] = prompt_version

    # 4. 初始化pipeline
    pipeline = PostCorrectionPipeline()
    pipeline.warmup()

    # 5. 解析layers选择
    layer_nums = [int(x.split("-")[0]) for x in layers]
    enable_semantic = 4 in layer_nums
    layer_nums = [x for x in layer_nums if x != 4]
    if not layer_nums:
        layer_nums = [1, 2, 3]

    # 6. 运行pipeline（一次性计算，但分步展示）
    original_html = diff_to_html(original_text, original_text, title="ASR原始输出")
    yield original_html, "", "", "", "", "", "", "", "", "", "Step 1-4: 纠错处理中（含大模型调用，请稍候）..."

    result = pipeline.run(
        original_text,
        layers=layer_nums,
        enable_semantic=enable_semantic,
        semantic_mode=semantic_mode,
    )

    # 7. 提取各层输出
    layer1_text = result.layer_outputs.get("layer1", original_text)
    layer2_text = result.layer_outputs.get("layer2", layer1_text)
    layer3_text = result.layer_outputs.get("layer3", layer2_text)
    layer4_text = result.layer_outputs.get("layer4", layer3_text)
    final_text = result.corrected

    # 8. 获取当前Prompt
    system_prompt, user_template = get_prompts(prompt_version)

    # 9. 构造实际发送的user prompt示例
    changes_section = ""
    if result.details:
        changes_lines = []
        for detail in result.details:
            for c in detail.changes:
                changes_lines.append(f"- {c.get('type', '修正')}: {c.get('before', '')} → {c.get('after', '')}")
        if changes_lines:
            changes_section = "【已确认的修改】\n" + "\n".join(changes_lines) + "\n\n"

    user_prompt_example = user_template.format(
        original_text=original_text,
        layer3_text=layer3_text,
        changes_section=changes_section,
    )

    config_str = f"语义模式: {semantic_mode} | Prompt: {prompt_version} | 启用层级: {', '.join(layers) if layers else '1-3'}"

    # 10. 分步流式展示
    empty = ""
    yield original_html, empty, empty, empty, empty, empty, empty, config_str, system_prompt, user_prompt_example, "Step 0/4: ASR原始输出 完成"
    time.sleep(0.3)

    layer1_html = diff_to_html(original_text, layer1_text, title="Layer 1 文本预处理")
    yield original_html, layer1_html, empty, empty, empty, empty, empty, config_str, system_prompt, user_prompt_example, "Step 1/4: 文本预处理 完成"
    time.sleep(0.3)

    layer2_html = diff_to_html(layer1_text, layer2_text, title="Layer 2 词典纠错")
    yield original_html, layer1_html, layer2_html, empty, empty, empty, empty, config_str, system_prompt, user_prompt_example, "Step 2/4: 词典纠错 完成"
    time.sleep(0.3)

    layer3_html = diff_to_html(layer2_text, layer3_text, title="Layer 3 上下文纠错")
    yield original_html, layer1_html, layer2_html, layer3_html, empty, empty, empty, config_str, system_prompt, user_prompt_example, "Step 3/4: 上下文纠错 完成"
    time.sleep(0.3)

    layer4_html = diff_to_html(layer3_text, layer4_text, title="Layer 4 语义精修")
    final_html = diff_to_html(original_text, final_text, title="最终输出（相对原始ASR的总修改）")
    correct_html = correct_diff_html(final_text, correct_text)
    yield original_html, layer1_html, layer2_html, layer3_html, layer4_html, final_html, correct_html, config_str, system_prompt, user_prompt_example, "Step 4/4: 语义精修 完成"


def create_demo() -> gr.Blocks:
    """创建Gradio界面."""

    with gr.Blocks(title="铁路调度ASR后处理纠错系统演示") as demo:
        gr.Markdown("# 铁路调度ASR后处理纠错系统演示")
        gr.Markdown("支持文本输入和音频上传两种方式，可实时查看每一层改写结果和当前使用的大模型Prompt。")

        with gr.Row():
            # 左侧：输入和配置
            with gr.Column(scale=1):
                gr.Markdown("## 输入")
                input_mode = gr.Radio(
                    choices=["文本输入", "音频上传"],
                    value="文本输入",
                    label="ASR输入方式",
                )

                text_input = gr.Textbox(
                    label="ASR原始文本",
                    placeholder="请输入ASR识别结果，例如：十八号道差开通反位",
                    lines=3,
                    visible=True,
                )

                correct_text = gr.Textbox(
                    label="正确答案（可选，用于对比高亮）",
                    placeholder="请输入标准答案，例如：18号道岔开通反位",
                    lines=2,
                    value="",
                )

                audio_input = gr.Audio(
                    label="上传音频文件",
                    type="filepath",
                    visible=False,
                )

                def toggle_input(mode):
                    return {
                        text_input: gr.update(visible=(mode == "文本输入")),
                        audio_input: gr.update(visible=(mode == "音频上传")),
                    }

                input_mode.change(toggle_input, inputs=input_mode, outputs=[text_input, audio_input])

                gr.Markdown("### ASR来源")
                asr_source = gr.Radio(
                    choices=["本地ASR(VibeVoice)", "qwen3-asr (vLLM)", "ElevenLabs ASR"],
                    value="qwen3-asr (vLLM)",
                    label="选择ASR服务",
                )

                with gr.Accordion("ASR服务配置（可手动修改）", open=False):
                    asr_base_url = gr.Textbox(label="ASR Base URL", value=LOCAL_ASR["base_url"])
                    asr_model = gr.Textbox(label="ASR Model", value=LOCAL_ASR["model"])
                    asr_api_key = gr.Textbox(label="ASR API Key", value=LOCAL_ASR["api_key"], type="password")

                def update_asr_config(source):
                    cfg = ASR_SOURCES.get(source, LOCAL_ASR)
                    return {
                        asr_base_url: gr.update(value=cfg["base_url"]),
                        asr_model: gr.update(value=cfg["model"]),
                        asr_api_key: gr.update(value=cfg["api_key"]),
                    }

                asr_source.change(update_asr_config, inputs=asr_source, outputs=[asr_base_url, asr_model, asr_api_key])

                audio_enhance = gr.Radio(
                    choices=list(_AUDIO_ENHANCE_OPTIONS.keys()),
                    value=_DEFAULT_AUDIO_ENHANCE,
                    label="音频增强/降噪（ASR 输入前）",
                )

                gr.Markdown("## 配置")
                hotwords_text = gr.Textbox(
                    label="热词编辑（逗号或换行分隔）",
                    placeholder="例如：道岔, 无表示, 销记, 总人解",
                    lines=3,
                    value="",
                )

                layers = gr.CheckboxGroup(
                    choices=["1-预处理", "2-词典纠错", "3-上下文纠错", "4-语义精修"],
                    value=["1-预处理", "2-词典纠错", "3-上下文纠错", "4-语义精修"],
                    label="启用层级",
                )

                semantic_mode = gr.Radio(
                    choices=["baseline", "rag", "harness"],
                    value="rag",
                    label="Layer 4 语义精修模式",
                )

                prompt_version = gr.Radio(
                    choices=["v1", "v2"],
                    value="v2",
                    label="Prompt版本（v2为铁路专家提示词，推荐RAG+v2）",
                )

                run_btn = gr.Button("开始纠错", variant="primary")

                gr.Markdown("### 提示")
                gr.Markdown("- v2 Prompt配合RAG模式效果最优\n- 热词修改后会覆盖data/lexicon/hotwords.json，演示结束后会自动恢复")

                with gr.Accordion("📖 各层方法说明", open=False):
                    gr.Markdown("""
                    **Step 0: ASR转写** — 将音频转为文本（本地VibeVoice或ElevenLabs）

                    **Step 1: 文本预处理** — 繁简转换、全半角规范化、标点初步清理

                    **Step 2: 词典纠错** — 基于配置化词典（同音词、数字格式、术语别名）进行规则替换

                    **Step 3: 上下文纠错** — 基于N-gram和上下文模式，处理连续词、局部搭配

                    **Step 4: 语义精修** — 大模型做最终语义校验，可选模式：
                    - **baseline**：单次LLM，通用提示词
                    - **rag**：检索增强 + 术语工具 + v2铁路专家提示词（推荐）
                    - **harness**：多策略竞争 + 裁判选择，效果最优但稍慢

                    **Prompt版本**：
                    - **v1**：通用ASR纠错提示词
                    - **v2**：铁路行车作业术语规范化专家提示词（含术语定义、推理铁律、Few-shot示例）
                    """)

            # 右侧：结果展示
            with gr.Column(scale=2):
                gr.Markdown("## 分步改写结果（绿色=新增/修改，红色删除线=删除/替换）")

                status_text = gr.Textbox(label="运行状态", interactive=False, value="等待开始...")
                original_output = gr.HTML(label="Step 0: ASR原始输出")
                layer1_output = gr.HTML(label="Step 1: Layer 1 文本预处理")
                layer2_output = gr.HTML(label="Step 2: Layer 2 词典纠错")
                layer3_output = gr.HTML(label="Step 3: Layer 3 上下文纠错")
                layer4_output = gr.HTML(label="Step 4: Layer 4 语义精修")
                final_output = gr.HTML(label="最终输出（相对ASR原文的总修改）")
                correct_output = gr.HTML(label="正确答案对比")

                config_info = gr.Textbox(label="当前配置", interactive=False)

                with gr.Accordion("当前大模型Prompt", open=False):
                    system_prompt_box = gr.Textbox(label="System Prompt", lines=20, interactive=False)
                    user_prompt_box = gr.Textbox(label="User Prompt示例", lines=10, interactive=False)

        # 绑定运行按钮
        run_btn.click(
            fn=process_correction,
            inputs=[
                input_mode,
                text_input,
                audio_input,
                asr_source,
                asr_base_url,
                asr_model,
                asr_api_key,
                hotwords_text,
                layers,
                semantic_mode,
                prompt_version,
                correct_text,
                audio_enhance,
            ],
            outputs=[
                original_output,
                layer1_output,
                layer2_output,
                layer3_output,
                layer4_output,
                final_output,
                correct_output,
                config_info,
                system_prompt_box,
                user_prompt_box,
                status_text,
            ],
        )

        # 示例
        gr.Examples(
            examples=[
                ["文本输入", "十八号道差开通反位，信号好了", None, "本地ASR(VibeVoice)", "道岔,无表示,销记", ["1-预处理", "2-词典纠错", "3-上下文纠错", "4-语义精修"], "rag", "v2", "18号道岔开通反位，信号好了", "不处理"],
                ["文本输入", "B站公务消记线路设备正常，电务登记十八号道岔五表示", None, "本地ASR(VibeVoice)", "工务,销记,无表示", ["1-预处理", "2-词典纠错", "3-上下文纠错", "4-语义精修"], "rag", "v2", "B站工务销记线路设备正常，电务登记18号道岔无表示", "不处理"],
                ["文本输入", "点击送人节按钮，进入解锁好了", None, "本地ASR(VibeVoice)", "总人解,进路", ["1-预处理", "2-词典纠错", "3-上下文纠错", "4-语义精修"], "rag", "v2", "点击总人解按钮，进路解锁好了", "不处理"],
                ["音频上传", "", None, "ElevenLabs ASR", "总人解,进路", ["1-预处理", "2-词典纠错", "3-上下文纠错", "4-语义精修"], "rag", "v2", "", "不处理"],
            ],
            inputs=[input_mode, text_input, audio_input, asr_source, hotwords_text, layers, semantic_mode, prompt_version, correct_text, audio_enhance],
            label="快速示例",
        )

    return demo


def main():
    """主入口."""
    # 启动时备份热词
    save_hotwords("")

    demo = create_demo()

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=True,  # 生成公网链接，便于演示
            show_error=True,
            quiet=False,
        )
    finally:
        # 关闭时恢复热词
        restore_hotwords()


if __name__ == "__main__":
    main()
