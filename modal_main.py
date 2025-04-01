import modal 

import train_finetune_accelerate

restart_tracker_dict = modal.Dict.from_name(
    "restart-tracker", create_if_missing=True
)

def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count

training_image = modal.Image.debian_slim(
    ).run_commands(["apt-get update", "apt-get install -y git"]
    ).pip_install_from_requirements(
    "requirements.txt"
    ).env({"HF_HOME": "/hf_cache"}
    ).add_local_dir(
    "Configs", remote_path="/root/Configs"
    ).add_local_dir(
    "Utils", remote_path="/root/Utils"
    )

app = modal.App(name="test", image=training_image)
style_tts2_volume = modal.Volume.from_name("style_tts2", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf_cache", create_if_missing=True)

@app.function(
    volumes={"/style_tts2": style_tts2_volume, "/hf_cache": hf_cache_volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
    gpu="H100",
    timeout = 86400)
def train_main():
    _ = track_restarts(restart_tracker_dict)
    train_finetune_accelerate.main({"config_path": "/root/Configs/config_ft.yml", "run_name": "modal_style_tts2"}) 

@app.local_entrypoint()
def main():
    train_main.remote()