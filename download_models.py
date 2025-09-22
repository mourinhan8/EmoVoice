from huggingface_hub import hf_hub_download
import os

hf_model_paths = {
    "llm_path": "Qwen/Qwen2.5-0.5B",
    "codec_path": "FunAudioLLM/CosyVoice-300M-SFT", 
    "ckpt_path": "yhaha/EmoVoice"
}

def download_models() -> str:
    """Download models from Hugging Face Hub"""
    try:
        print("Downloading models from Hugging Face Hub...")
        
        # Download Qwen2.5-0.5B
        print("Downloading Qwen2.5-0.5B...")
        qwen_path = hf_hub_download(
            repo_id=hf_model_paths["llm_path"],
            filename="config.json",
            cache_dir="./models_hf"
        )
        qwen_dir = os.path.dirname(qwen_path)
        
        # Download CosyVoice-300M-SFT
        print("Downloading CosyVoice-300M-SFT...")
        cosyvoice_path = hf_hub_download(
            repo_id=hf_model_paths["codec_path"],
            filename="configuration.json", 
            cache_dir="./models_hf"
        )
        cosyvoice_dir = os.path.dirname(cosyvoice_path)
        
        # Download EmoVoice checkpoint
        print("Downloading EmoVoice checkpoint...")
        emovoice_path = hf_hub_download(
            repo_id=hf_model_paths["ckpt_path"],
            filename="EmoVoice.pt",
            cache_dir="./models_hf"
        )
        print(f"✅ Models downloaded successfully!\nQwen: {qwen_dir}\nCosyVoice: {cosyvoice_dir}\nEmoVoice: {emovoice_path}")
        
    except Exception as e:
        print(f"Error downloading models: {str(e)}")

if __name__ == "__main__":
    download_models()