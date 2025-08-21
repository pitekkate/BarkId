# app.py
from transformers import VitsModel, AutoTokenizer, pipeline
import torch
import scipy
import numpy as np
import tempfile
import gradio as gr

# --- Muat Model MMS-TTS (Suara Alami Bahasa Indonesia) ---
print("📥 Memuat model facebook/mms-tts-ind...")
mms_model = VitsModel.from_pretrained("facebook/mms-tts-ind")
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")
mms_sampling_rate = mms_model.config.sampling_rate

# --- Muat Model Bark (Efek Suara & Musik) ---
print("📥 Memuat pipeline suno/bark...")
bark_pipe = pipeline("text-to-speech", "suno/bark")

# Voice preset Bark
BARK_PRESETS = [
    "v2/en_speaker_6",
    "v2/en_speaker_0",
    "v2/en_speaker_9",
]

def generate_mms(text):
    if not text.strip():
        return None
    inputs = mms_tokenizer(text.strip(), return_tensors="pt")
    with torch.no_grad():
        output = mms_model(**inputs).waveform
    audio = output.numpy().squeeze()
    audio = audio / max(0.01, np.max(np.abs(audio))) * 0.9
    return mms_sampling_rate, audio

def generate_bark(text, preset, add_bg):
    try:
        speech = bark_pipe(
            text,
            forward_params={"do_sample": True, "temperature": 0.7},
            generate_kwargs={"history_prompt": preset}
        )
        sr = speech["sampling_rate"]
        audio = speech["audio"]
        if add_bg:
            bg = np.random.normal(0, 0.003, len(audio))
            bg = np.tile(bg, len(audio)//len(bg) + 1)[:len(audio)]
            audio = audio * 0.9 + bg * 0.1
            audio = np.clip(audio, -1.0, 1.0)
        return sr, audio
    except Exception as e:
        print(f"Bark error: {e}")
        return None, None

def tts_engine(text, model_choice, bark_preset, add_background=False):
    if not text.strip():
        return None

    sr, audio = None, None
    if model_choice == "MMS-TTS (Suara Alami)":
        sr, audio = generate_mms(text)
    elif model_choice == "Bark (Efek & Musik)":
        sr, audio = generate_bark(text, bark_preset, add_background)

    if audio is None:
        return None

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    scipy.io.wavfile.write(temp_file.name, rate=sr, data=audio)
    return temp_file.name

# --- Antarmuka Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🇮🇩 TTS Indonesia: MMS-TTS + Bark
    Aplikasi ini menggabungkan dua model terbaik:
    - ✅ **MMS-TTS**: Suara alami bahasa Indonesia
    - ✅ **Bark**: Efek suara, musik, tawa (`[laughs]`, `[music]`)
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Teks (dalam bahasa Indonesia)",
                placeholder="Contoh: Saya suka makan sate [music] sambil tertawa [laughs]",
                lines=5
            )
            model_choice = gr.Radio(
                ["MMS-TTS (Suara Alami)", "Bark (Efek & Musik)"],
                value="MMS-TTS (Suara Alami)",
                label="Pilih Model"
            )
            with gr.Group(visible=False) as bark_options:
                bark_preset = gr.Dropdown(BARK_PRESETS, label="Suara", value="v2/en_speaker_6")
                add_bg = gr.Checkbox(label="Tambahkan nuansa musik latar?", value=True)
            btn = gr.Button("🔊 Hasilkan Audio")

        with gr.Column():
            output_audio = gr.Audio(label="Hasil Audio")

    model_choice.change(
        fn=lambda choice: gr.update(visible=choice=="Bark (Efek & Musik)"),
        inputs=model_choice,
        outputs=bark_options
    )

    btn.click(
        fn=tts_engine,
        inputs=[text_input, model_choice, bark_preset, add_bg],
        outputs=output_audio
    )

    gr.Markdown("""
    > ℹ️ Model:  
    > - [facebook/mms-tts-ind](https://huggingface.co/facebook/mms-tts-ind) – CC-BY-NC-4.0  
    > - [suno/bark](https://huggingface.co/suno/bark) – CC-BY-NC-4.0
    """)

if __name__ == "__main__":
    demo.launch()
