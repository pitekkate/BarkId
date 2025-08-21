# app.py
import gradio as gr
import torch
import scipy
import tempfile
import numpy as np
from transformers import pipeline, VitsModel, AutoTokenizer

# --- Muat Model ---
# Model 1: Bark (untuk ekspresi dan musik)
bark_pipe = pipeline("text-to-speech", "suno/bark")

# Model 2: MMS-TTS untuk Indonesia (suara alami)
mms_model = VitsModel.from_pretrained("facebook/mms-tts-ind")
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")

# Daftar voice preset Bark
BARK_PRESETS = [
    "v2/en_speaker_6",
    "v2/en_speaker_0",
    "v2/en_speaker_9",
]

def generate_bark(text, preset):
    try:
        speech = bark_pipe(
            text,
            forward_params={"do_sample": True, "temperature": 0.7},
            generate_kwargs={"history_prompt": preset}
        )
        audio = speech["audio"]
        sr = speech["sampling_rate"]
    except Exception as e:
        print(f"Bark error: {e}")
        return None
    return sr, audio

def generate_mms(text):
    try:
        inputs = mms_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = mms_model(**inputs).waveform
        audio = waveform.numpy().squeeze()
        sr = mms_model.config.sampling_rate
    except Exception as e:
        print(f"MMS error: {e}")
        return None
    return sr, audio

def tts_engine(text, model_choice, bark_preset, add_background=False):
    if not text.strip():
        return None

    text = text.strip() + " ."

    if model_choice == "Bark (dengan efek suara)":
        result = generate_bark(text, bark_preset)
        if result is None:
            return None
        sr, audio = result

        # Tambahkan nuansa musik latar (opsional)
        if add_background and len(audio) > 0:
            bg = np.random.normal(0, 0.003, len(audio))
            audio = audio * 0.9 + bg * 0.1
            audio = np.clip(audio, -1.0, 1.0)

    elif model_choice == "MMS-TTS (suara alami)":
        result = generate_mms(text)
        if result is None:
            return None
        sr, audio = result
    else:
        return None

    # Simpan sementara
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    scipy.io.wavfile.write(temp_file.name, rate=sr, data=audio)
    return temp_file.name

# --- Antarmuka Gradio ---
with gr.Blocks(theme=gr.themes.Soft(), title="🇮🇩 TTS Indonesia") as demo:
    gr.Markdown("""
    # 🇮🇩 TTS Indonesia: Bark vs MMS-TTS
    Aplikasi ini membandingkan dua model TTS untuk bahasa Indonesia:
    
    - **Bark**: Suara ekspresif dengan tawa, musik, dan efek suara (`[laughs]`, `[music]`, `♪`)
    - **MMS-TTS**: Suara alami dan jernih, khusus dilatih untuk bahasa Indonesia
    
    Pilih model dan dengarkan perbedaannya!
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Teks dalam Bahasa Indonesia",
                placeholder="Contoh: Saya suka makan sate di pinggir jalan sambil mendengarkan musik [music] dan tertawa bersama teman.",
                lines=5
            )

            model_choice = gr.Radio(
                choices=[
                    "Bark (dengan efek suara)",
                    "MMS-TTS (suara alami)"
                ],
                value="Bark (dengan efek suara)",
                label="Pilih Model"
            )

            with gr.Group(visible=True) as bark_options:
                bark_preset = gr.Dropdown(BARK_PRESETS, label="Suara", value="v2/en_speaker_6")
                add_bg = gr.Checkbox(label="Tambahkan nuansa musik latar?", value=True)

            btn = gr.Button("🔊 Hasilkan Audio")

        with gr.Column(scale=3):
            output_audio = gr.Audio(label="Hasil Audio", type="filepath")

    # Update opsi berdasarkan pilihan model
    def show_bark_options(choice):
        return gr.update(visible=choice == "Bark (dengan efek suara)")

    model_choice.change(
        fn=show_bark_options,
        inputs=model_choice,
        outputs=bark_options
    )

    btn.click(
        fn=tts_engine,
        inputs=[text_input, model_choice, bark_preset, add_bg],
        outputs=output_audio
    )

    gr.Markdown("""
    ### 📌 Panduan Penggunaan
    - **Gunakan `[music]` atau `♪`** hanya di **Bark** untuk efek musik.
    - **MMS-TTS** tidak mendukung efek suara, tapi suaranya lebih alami.
    - Cocok untuk:
      - Konten YouTube (Bark)
      - Audiobook, e-learning (MMS-TTS)
      - Iklan lokal (kombinasi keduanya)

    > ℹ️ Model:  
    > - [suno/bark](https://huggingface.co/suno/bark) – CC-BY-NC-4.0  
    > - [facebook/mms-tts-ind](https://huggingface.co/facebook/mms-tts-ind) – CC-BY-NC-4.0
    """)

if __name__ == "__main__":
    demo.launch()
