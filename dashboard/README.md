# AdvGuard Defence Intelligence Lab

## Discovered project interfaces

- Victim and protected classifiers use the existing `model.create_resnet18_cifar10()` ResNet-18: a 3×3 CIFAR stem, identity max-pool, and 10-class head. `utils.victim_model.load_victim_model` defines the project's existing state-dict loading conventions.
- `attacks.fgsm.fgsm_attack`, `attacks.pgd.pgd_attack`, and `attacks.cw.cw_l2_attack` accept normalized model inputs and return normalized adversarial inputs. Perturbation limits and image clipping are performed in pixel space.
- CIFAR-10 normalization is `(0.4914, 0.4822, 0.4465)` / `(0.2470, 0.2435, 0.2616)`. Uploaded images are converted to RGB, resized to 32×32, then normalized for classifier and detector inference.
- `detector.AdversarialDetector` takes normalized `[N,3,32,32]` input and returns one raw binary logit per image. No trained detector checkpoint exists in the inspected project; detector output is therefore disabled and never synthesized. Existing detector scripts mention `model/adversarial_detector_final.pth` as a default, but that file is absent.
- `purification.PurificationNet` takes and returns RGB pixel-space tensors in `[0,1]`. It is loaded from `model/purifier_best.pth`; its metadata defaults to 48 channels and 4 residual blocks.
- Available checkpoints: `resnet18_cifar10.pth`, `protected_resnet18_baseline.pth`, `protected_resnet18_continued_best.pth`, and `purifier_best.pth`. All were left untouched. No data in the local CIFAR-10 test-set cache was found during inspection.

## Run locally (PowerShell)

From the repository root, install backend dependencies into the existing virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\dashboard\requirements.txt
.\venv\Scripts\python.exe -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Inference stays on the local machine. The API listens on loopback only. CPU is supported; CUDA is selected automatically when available. FGSM, PGD, and C&W attacks execute in a worker thread so they do not block the FastAPI event loop.

The frontend development dependencies include React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide, and Framer Motion. `npm run build` creates the frontend production bundle.

## Scientific behavior and limits

- A successful untargeted attack is reported only when the victim's actual top prediction changes from its clean prediction. This is a prediction-change measure; no ground-truth label is inferred for an uploaded image.
- Confidence is shown separately from accuracy. Classification correctness cannot be scored without a supplied ground-truth label.
- Attack settings are returned and stored with each run. Attacks are generated against the victim classifier and are non-adaptive evaluations.
- Detector inference requires the missing trained checkpoint and remains unavailable until it is supplied. The purifier and protected classifier can load independently.
- Session experiment history is held in the local API process memory and cleared when that process restarts. JSON exports include the rendered experiment images; CSV exports contain scalar summaries.
