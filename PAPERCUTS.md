# Papercuts

Small, non-blocking friction hit while working in this repo. One entry per papercut, newest at the bottom.
Read this before starting work; append to it the moment something gets in your way.
Not for accomplishments (LOG.md) or real bugs (issue tracker).

## 2026-11-04 · modal/gpu · Modal GPU access requires payment method on file
- **Model:** gemini-3.8-flash
- **Doing:** Running `modal run qwen_image_turbo.py` with `gpu="L40S"`.
- **Hit:** `InvalidError('Please add a payment method to use L40S GPU functions.')`. Modal requires a credit card on file in the dashboard to unlock GPU execution, even when using starter free credits.
- **Workaround:** Add a payment method in https://modal.com/settings/billing before running GPU functions.
- **Likely cause / fix:** Policy gate on Modal accounts to prevent abuse on GPU clusters.
- **Status:** Fixed (2026-11-04) — Payment method added to Modal account; GPU functions are now approved.

## 2026-11-04 · modal/deps · Qwen3VLProcessor requires torchvision and sentencepiece
- **Model:** gemini-3.8-flash
- **Doing:** Loading QwenImage21Pipeline on Modal.
- **Hit:** `ImportError: Qwen3VLVideoProcessor requires the Torchvision library but it was not found in your environment.`
- **Workaround:** Add `"torchvision"` and `"sentencepiece"` to `image.pip_install(...)`.
- **Likely cause / fix:** `transformers` Qwen3-VL processor imports video/image processing submodules that depend on `torchvision`.
- **Status:** Fixed (2026-11-04) — Added to pip_install in qwen_image_turbo.py.
