"""Qwen-Image-2.1 + Viggle Turbo LoRA on Modal.

Test once:   modal run qwen_image_turbo.py --prompt "a red fox in snow, photo"
Deploy API:  modal deploy qwen_image_turbo.py
"""
import io
import time
import modal

app = modal.App("qwen-image-turbo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        "sentencepiece",
        "transformers>=5.17",
        "accelerate",
        "peft",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "pillow",
        "git+https://github.com/huggingface/diffusers.git",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache"})
)

# Keeps downloaded weights between runs so you only download once
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

BASE_MODEL_ID = "Qwen/Qwen-Image-2.1"
LORA_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
LORA_FILE = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors"

STEPS = 6
SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/cache": cache},
    timeout=1800,
    scaledown_window=120,  # shut the GPU down after 2 idle minutes
)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline

        print("Loading Qwen-Image-2.1 base pipeline...")
        try:
            self.pipe = QwenImage21Pipeline.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.bfloat16,
                local_files_only=True,
            )
        except Exception:
            self.pipe = QwenImage21Pipeline.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.bfloat16,
            )

        print(f"Loading Viggle Turbo LoRA ({LORA_FILE})...")
        try:
            self.pipe.load_lora_weights(
                LORA_REPO,
                weight_name=LORA_FILE,
                adapter_name="turbo",
                local_files_only=True,
            )
        except Exception:
            self.pipe.load_lora_weights(
                LORA_REPO,
                weight_name=LORA_FILE,
                adapter_name="turbo",
            )

        print("Configuring FlowMatchEulerDiscreteScheduler...")
        try:
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                LORA_REPO,
                subfolder="scheduler",
                local_files_only=True,
            )
        except Exception:
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                LORA_REPO,
                subfolder="scheduler",
            )

        print("Moving pipeline to CUDA...")
        self.pipe.to("cuda")
        cache.commit()
        print("Model loaded and cached successfully.")

    def _generate(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0) -> bytes:
        import torch

        print(f"Generating image for: '{prompt}' ({width}x{height}, seed={seed})")
        img = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=STEPS,
            sigmas=SIGMAS,
            true_cfg_scale=1.0,  # turbo LoRA was distilled without CFG
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    def generate(self, prompt: str, seed: int = 0) -> bytes:
        return self._generate(prompt, seed=seed)

    @modal.fastapi_endpoint(method="GET")
    def web(self, prompt: str, seed: int = 0):
        from fastapi import Response

        return Response(self._generate(prompt, seed=seed), media_type="image/png")


@app.local_entrypoint()
def main(
    prompt: str = "a red fox in fresh snow, golden hour photo",
    seed: int = 0,
    output: str = "output.png",
    interactive: bool = False,
):
    model = Model()
    if interactive:
        print("\n⚡ Interactive mode active — container will stay warm in GPU memory between prompts.")
        print("Type a prompt and press Enter. Type 'exit' or 'q' to stop.\n")
        idx = 1
        while True:
            try:
                user_prompt = input("Enter prompt > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            if not user_prompt or user_prompt.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            filename = f"output_{idx}.png" if output == "output.png" else f"{idx}_{output}"
            print(f"Generating image {idx}...")
            t0 = time.time()
            png = model.generate.remote(user_prompt, seed)
            with open(filename, "wb") as f:
                f.write(png)
            print(f"✓ Saved {filename} in {time.time() - t0:.2f}s\n")
            idx += 1
    else:
        print(f"Triggering remote generation with prompt: '{prompt}' (seed={seed})...")
        t0 = time.time()
        png = model.generate.remote(prompt, seed)
        with open(output, "wb") as f:
            f.write(png)
        print(f"✓ Saved {output} in {time.time() - t0:.2f}s")
