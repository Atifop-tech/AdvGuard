# AdvGuard

Demonstration of **FGSM, PGD, and Carlini & Wagner (C&W)** adversarial attacks on a ResNet-18 victim classifier trained on CIFAR-10.

The project explores how small changes to input images affect classification. It includes classifier training, a visual FGSM demonstration, PGD and C&W evaluation scripts, a trained checkpoint, and recorded evaluation metrics. Defense and detector implementations are not yet included.

## Implemented features

| Component | Description |
| --- | --- |
| Victim model | ResNet-18 adapted to 32 × 32 images and 10 CIFAR-10 classes |
| FGSM | Single gradient-sign step, with an original/adversarial image comparison |
| PGD | Iterative L-infinity attack with projection and optional random initialization |
| C&W | Untargeted L2 optimization using a tanh parameterization and a fixed `c` value |
| Evaluation | Clean accuracy, adversarial accuracy, attack success rate, confidence, and perturbation magnitude exported to CSV |

The model uses a 3 × 3 initial convolution, replaces the initial max-pooling layer with an identity operation, and predicts 10 classes. Scripts select CUDA when available and otherwise use the CPU.

## Project structure

```text
AdvGuard/
├── Attacks/                    # PGD and C&W implementations; see capitalization note below
│   ├── __init__.py
│   ├── pgd.py
│   └── cw.py
├── model/
│   ├── __init__.py             # Model factory
│   └── resnet18_cifar10.pth     # Trained victim checkpoint
├── results/
│   ├── pgd_metrics.csv
│   └── cw_metrics.csv
├── Checkpoints/                # Placeholder
├── Defence/                    # Placeholder
├── Detector/                   # Placeholder
├── Model.py                    # Empty placeholder
├── train_classifier.py
├── test_attack.py              # Visual FGSM demo
├── evaluate_pgd.py
├── evaluate_cw.py
└── requirements.txt
```

Downloaded datasets and the local Python environment are excluded from Git. CIFAR-10 is downloaded automatically when a script needs it.

## Setup

Install Git and Python. Python 3.12 matches the original development environment. Run these commands in a terminal:

```bash
git clone https://github.com/Atifop-tech/AdvGuard.git
cd AdvGuard
python -m venv venv
```

Activate the environment using the command for your terminal:

| Terminal | Command |
| --- | --- |
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `venv\Scripts\activate.bat` |
| macOS / Linux | `source venv/bin/activate` |

On systems where Python is available as `python3`, use `python3 -m venv venv` to create the environment.

Install dependencies after activation:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies include PyTorch, torchvision, torchattacks, NumPy, Matplotlib, pandas, scikit-learn, Pillow, tqdm, and Streamlit. Package versions are currently unpinned. The repository does not yet contain a Streamlit application.

Run all commands below from the repository root. Internet access is needed for dependency installation and the first dataset download. CPU execution is supported, but attack evaluation and training can take longer than on a GPU.

### Current path compatibility notes

1. **Attack package capitalization:** Git currently stores the package as `Attacks/`, while the evaluation scripts import `attacks`. On a case-sensitive filesystem, rename the directory before running the evaluations:

   ```bash
   mv Attacks attacks
   ```

2. **FGSM checkpoint path:** In `test_attack.py`, replace the existing `MODEL_PATH` assignment with:

   ```python
   MODEL_PATH = "./model/resnet18_cifar10.pth"
   ```

   The checked-in checkpoint is in `model/`; the script currently refers to `models/`.

3. **Training on Windows:** If training reports multiprocessing startup errors, set both `num_workers=2` entries in `train_classifier.py` to `num_workers=0`. The evaluation commands below already use zero workers for a straightforward setup.

## Run the FGSM demonstration

After correcting `MODEL_PATH` as described above:

```bash
python test_attack.py
```

The script finds a CIFAR-10 test image that the victim model classifies correctly, applies FGSM with `epsilon = 8 / 255`, and prints the predictions and confidence before and after the attack. A Matplotlib window displays the two images. A successful attack changes the prediction away from the true class; success is not guaranteed for every image.

## Evaluate PGD

Evaluate the first 1,000 test images using the default `epsilon = 8 / 255`, `alpha = 2 / 255`, 10 steps, and a random start:

```bash
python evaluate_pgd.py --checkpoint model/resnet18_cifar10.pth --max-samples 1000 --num-workers 0 --output results/pgd_attack_metrics.csv
```

Omit `--max-samples` to evaluate the full test set. Additional options include `--epsilon`, `--alpha`, `--steps`, `--batch-size`, `--seed`, `--data-dir`, and `--no-random-start`.

For a zero-perturbation baseline:

```bash
python evaluate_pgd.py --checkpoint model/resnet18_cifar10.pth --epsilon 0 --max-samples 1000 --num-workers 0 --output results/pgd_baseline_metrics.csv
```

## Evaluate C&W

Evaluate the first 100 test images using `c = 1.0`, `kappa = 0.0`, 50 optimization steps, and learning rate `0.01`:

```bash
python evaluate_cw.py --checkpoint model/resnet18_cifar10.pth --max-samples 100 --num-workers 0 --output results/cw_attack_metrics.csv
```

Additional options include `--c`, `--kappa`, `--steps`, `--learning-rate`, `--batch-size`, `--seed`, and `--data-dir`. The default sample limit is 100; use `--max-samples 10000` for the full CIFAR-10 test set.

This implementation uses a fixed `c` value, without a binary search over `c`. If no successful candidate is found for an image, it returns the final optimized candidate.

To inspect all evaluation options:

```bash
python evaluate_pgd.py --help
python evaluate_cw.py --help
```

## Recorded results

The following values come from the CSV files committed in `results/`. They are saved runs, not results from a fresh execution of the commands above.

| Saved run | Images | Clean accuracy | Adversarial accuracy | Attack success rate | Mean perturbation |
| --- | ---: | ---: | ---: | ---: | ---: |
| PGD baseline (`epsilon = 0`) | 1,000 | 92.3% | 92.3% | 0% | L-infinity: 0 |
| C&W (`c = 1`, 50 steps) | 100 | 93.0% | 0% | 100% | L2: 0.3195 |

**The saved PGD run is a zero-perturbation baseline**, so it does not measure robustness against a nonzero PGD attack. The two runs also use different sample counts and should not be treated as a controlled comparison of attack strength.

- **Clean accuracy:** fraction of evaluated images classified correctly before the attack.
- **Adversarial accuracy:** fraction classified correctly after the attack.
- **Attack success rate:** fraction of initially correct predictions made incorrect by the attack.
- **Mean perturbation:** average pixel-space L-infinity distance for PGD or L2 distance for C&W.

CSV accuracy and success-rate values are stored as fractions; the table displays percentages. The scripts also save mean prediction confidence. Evaluation output files are overwritten when the same output path is reused.

## Train the victim model

The trained checkpoint is already included, so retraining is optional.

```bash
python train_classifier.py
```

Training defaults are 20 epochs, batch size 128, Adam with learning rate `0.001`, and cosine annealing. Training augmentation uses random cropping and horizontal flipping. Inputs use CIFAR-10 normalization with mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`.

The training script saves its best checkpoint to `models/resnet18_cifar10.pth`, which is separate from the included `model/resnet18_cifar10.pth`. To evaluate a newly trained checkpoint, pass `--checkpoint models/resnet18_cifar10.pth`; for the FGSM demo, update `MODEL_PATH` accordingly.

## Get updates

From the project directory:

```bash
git pull
python -m pip install -r requirements.txt
```

Commit or stash local edits before pulling if Git reports conflicting changes.
