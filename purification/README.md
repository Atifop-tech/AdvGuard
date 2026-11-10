# AdvGuard Purification Experiment

This branch adds a paired image-restoration baseline to the existing detector and protected-classifier experiments.

## Method

- Input: CIFAR-10 RGB image in pixel space [0, 1].
- Target: the clean source image associated with that adversarial sample.
- Network: residual CNN with GroupNorm and SiLU blocks. It predicts a bounded residual correction and starts as an identity mapping.
- Training pairs: the existing `train.pt` and `validation.pt` files, matched by `source_indices`. Every source contributes its adversarial image -> clean image pair and a clean -> clean identity pair.
- Loss: Smooth L1 reconstruction plus 0.10 times horizontal/vertical gradient L1 loss.
- Model selection: lowest validation reconstruction loss. No test-set-based checkpoint selection.

The purifier is deliberately classifier-independent. It does not use the victim's predicted class or true class as a training target.

## Before training

Run these commands from the repository root. Confirm the existing paired datasets are present and do not regenerate or overwrite the independent test set.

```bash
python -m pip install -r requirements.txt
python -m purification.train --train-dataset adversarial_dataset/train.pt --validation-dataset adversarial_dataset/validation.pt --output model/purifier_best.pth --history results/purifier_training_history.csv --epochs 40 --batch-size 32 --num-workers 0
```

On CPU, training may take a while. Do not change hyperparameters after examining test results.

## Evaluate once

```bash
python -m purification.evaluate --test-dataset adversarial_dataset/test.pt --purifier-checkpoint model/purifier_best.pth --victim-checkpoint model/resnet18_cifar10.pth --protected-checkpoint model/protected_resnet18_continued_best.pth --output-dir results/purification --batch-size 64 --num-workers 0
```

The evaluation writes per-sample predictions and a group summary for clean, FGSM, PGD, and C&W images. It compares each classifier on raw and purified inputs and reports image reconstruction MAE/MSE against the paired clean source.

## Interpretation and limitations

- Report clean accuracy before and after purification; a purifier that damages clean images is not a useful defence.
- Report attack-specific accuracy, not only pooled adversarial accuracy.
- Report reconstruction MAE/MSE alongside classifier accuracy.
- This is a **non-adaptive** evaluation: the attacks in the existing test tensor were generated against the original victim, not against the purifier-plus-classifier composition. It does not establish robustness to an adaptive attacker.
- The current C&W samples inherit the repository's fixed-c, 50-step C&W configuration. Results must be described with those attack settings and must not be compared as if they were a fully tuned C&W benchmark.
- Keep the test split untouched until the purifier checkpoint is selected on validation. Run the test evaluation once for the final report and preserve its CSV outputs.
- If the purifier reduces adversarial accuracy or harms clean accuracy, report that outcome. Do not tune against the test set.
