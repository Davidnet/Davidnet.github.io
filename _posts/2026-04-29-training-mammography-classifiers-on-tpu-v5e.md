---
title: "From scratch on TPU v5e: training mammography classifiers in JAX/Flax"
description: "Notes from running 15 deep-learning experiments on a Google Cloud v5litepod-8 TPU for my undergrad thesis on the CBIS-DDSM patch dataset. What worked, what didn't, and the surprises along the way."
date: 2026-04-29
categories: [blog]
layout: page
tags: [jax, flax, tpu, machine-learning, medical-imaging, thesis]
---

# From scratch on TPU v5e: training mammography classifiers in JAX/Flax

This is a write-up of the TPU portion of my undergraduate thesis (CM3070, *Computer Science Final Project*, University of London Goldsmiths). The thesis built two parallel pipelines for the same task: patch-level classification on the [Curated Breast Imaging Subset of DDSM](https://www.tensorflow.org/datasets/catalog/curated_breast_imaging_ddsm). Across both pipelines I trained 15 deep-learning experiments on a Google Cloud `v5litepod-8` TPU v5e accelerator using JAX/Flax. The whole project is on GitHub: [`CM3070-curated_breast_imaging_ddsm`](https://github.com/Davidnet/CM3070-curated_breast_imaging_ddsm).

The post focuses on the JAX/Flax/TPU pipeline. A separate PyTorch pipeline handled pretrained models and explainability, and that one is a story for another day.

## The setup

The dataset is `curated_breast_imaging_ddsm/patches:3.0.0`, distributed via TensorFlow Datasets. Each example is a 224 x 224 single-channel mammographic patch with one of five labels: `BACKGROUND`, `BENIGN_CALCIFICATION`, `BENIGN_MASS`, `MALIGNANT_CALCIFICATION`, `MALIGNANT_MASS`. The splits are fixed at 49,780 train / 5,580 validation / 9,770 test samples.

![Class distribution on the 5-class TPU task](/assets/img/blog/tpu/class_distribution_5class.png)

The class imbalance shapes everything that follows: BACKGROUND alone is 46.6% of the training set, while the two malignant classes together are 22.3%. Balanced accuracy and macro-F1 are the metrics that matter; raw accuracy can always be hacked by predicting BACKGROUND more often.

## Why TPU, and why JAX

This was a deliberate "learning by doing" choice. The PyTorch ecosystem is more mature for medical imaging, has [timm](https://github.com/huggingface/pytorch-image-models) for pretrained backbones, has [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for explainability, and has a clean ONNX export path. JAX has none of those out of the box. But JAX gives you `pmap`, `jit`, fully functional state, and a TPU programming model that is genuinely fun to write. On top of that, TPU v5e is significantly cheaper per FLOP than equivalent GPU options if you can stomach the rougher tooling.

So the TPU pipeline runs from random initialisation. No pretrained weights. The whole point was to see how far a from-scratch model can go on a constrained dataset, and what "training a model on a TPU pod" actually feels like end-to-end: provisioning, data on GCS, multi-host data parallelism, checkpointing, the lot.

## The pipeline

The training loop is built around `jax.pmap` on a `v5litepod-8` slice: 8 TPU v5e chips, one TensorCore per chip, 16 GB HBM per chip for 128 GB total. JAX sees 8 devices, and `pmap` shards the global batch across all of them. Mixed precision is bfloat16 throughout, AdamW + warmup-cosine, gradient norm clipped at 1.0, Orbax for checkpointing. A typical run config looks like this (the `final_best` ResNet34 recipe):

```yaml
# resnet34_gray model
stage_sizes: [3, 4, 6, 3]
widths: [64, 128, 256, 512]
in_channels: 1
dropout_rate: 0.1
stochastic_depth_rate: 0.05

# training
seed: 21
epochs: 35
batch_size: 256          # global, sharded 8-way across cores
learning_rate: 0.0007
warmup_epochs: 3
weight_decay: 0.0002
optimizer: adamw
loss_name: focal
class_weights: [0.5, 1.0, 1.0, 1.3, 1.3]
label_smoothing: 0.03
dropout_rate_override: 0.15
ema_decay: 0.999
use_bfloat16: true
grad_clip_norm: 1.0
selection_metric: macro_f1
```

The training step is a thin pmap closure with all-reduces at the right places. Gradients, batch-stats, and EMA updates are all `pmean`'d across the `batch` axis:

```python
def train_step(state, batch):
    dropout_rng, next_rng = jax.random.split(state.rng)
    dropout_rng = jax.random.fold_in(dropout_rng, jax.lax.axis_index("batch"))

    def loss_fn(params):
        variables = _state_variables(state) | {"params": params}
        logits, updates = model.apply(
            variables, batch["image"], train=True,
            rngs={"dropout": dropout_rng},
            mutable=["batch_stats"],
        )
        loss = classification_loss(logits, batch["label"], cfg, num_classes)
        return loss, (logits, updates["batch_stats"])

    (loss, (logits, new_bs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    new_state = state.apply_gradients(grads=grads).replace(
        batch_stats=jax.lax.pmean(new_bs, axis_name="batch"),
        rng=next_rng,
    )
    if cfg.ema_decay is not None:
        ema = jax.tree_util.tree_map(
            lambda o, n: cfg.ema_decay * o + (1.0 - cfg.ema_decay) * n,
            state.ema_params, new_state.params,
        )
        new_state = new_state.replace(ema_params=ema)
    return new_state, {"loss": loss}
```

Once you're comfortable with the functional state pattern, this style of training loop is honestly delightful. Every transformation is a pure function, and the `pmap` boundary makes data parallelism explicit instead of magical.

## What I trained

Five architectures, all from random initialisation:

- **SmallCNN**: a 4-block convnet, used only as a smoke test for the pipeline.
- **ResNet18 / ResNet34**: adapted for grayscale (5×5 stride-2 stem), GroupNorm/BatchNorm, stochastic depth.
- **ResNet50**: same family. Note that the Flax implementation uses basic blocks for all depths, which makes ResNet50 architecturally identical to ResNet34 here. (Lesson learned: always paper-trace the implementation, not the name.)
- **ViT-Tiny / ViT-Small**: 16×16 patch embedding, learnable positional embeddings, [CLS] token.

Across the five architectures plus a tuning campaign on ResNet34 plus a hyperparameter ablation, I shipped 15 TPU runs.

![Baseline architecture training curves](/assets/img/blog/tpu/dl_training_curves_baselines.png)

The convergence patterns split cleanly along architecture lines. ResNets descend smoothly. ViTs sit near chance for the first 5–10 epochs and then start learning. That is a now-familiar transformer warmup pattern that you have to budget compute for.

![Reconstructed warmup-cosine LR schedules](/assets/img/blog/tpu/dl_lr_schedules_tpu.png)

The schedule is the standard linear warmup → cosine decay; the figure above is reconstructed analytically from the configs because, embarrassingly, I had not logged the LR step-by-step during training. Lesson #1 for next time.

## Baseline results, 5-class test set

| Architecture | Params | Test Acc. | Test Macro-F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| ResNet18 | 11.2M | 0.410 | 0.231 | 0.662 |
| **ResNet34** | 21.3M | **0.580** | **0.485** | **0.800** |
| ResNet50* | 21.3M | 0.584 | 0.481 | 0.801 |
| ViT-Tiny  | 5.4M  | 0.496 | 0.337 | 0.706 |
| ViT-Small | 21.5M | 0.514 | 0.372 | 0.720 |

Two things stand out immediately:

1. **ResNets crush ViTs from scratch.** The ViTs don't have the inductive biases (locality, translation equivariance) that convolutions hand you for free, and on ~50k training images that gap is brutal. You can see it in the curves and you can see it in the final macro-F1: 11–15 pp behind ResNet34.

2. **ResNet18 collapses on the small classes.** Its 0.231 macro-F1 hides zero recall on `BENIGN_MASS` and `MALIGNANT_MASS`. The validation curves wobble for the entire 25 epochs without ever stabilising. At this image scale, a ResNet18 just doesn't have the capacity to resolve all five classes simultaneously.

## Tuning ResNet34: the slog

ResNet34 became the workhorse. I ran a small progression: cross-entropy baseline → focal loss only → focal + class weights + dropout + LR/WD tweaks → "final best".

![ResNet34 progression: baseline through final_best](/assets/img/blog/tpu/dl_resnet34_progression.png)

| Run | Loss | LR | Class Wt. | Test Macro-F1 |
|---|---|---|---|---:|
| baseline | CE | 5e-4 | none | 0.485 |
| focal_only | Focal | 5e-4 | none | 0.466 |
| tuned | Focal | 7e-4 | [0.5, 1, 1, 1.3, 1.3] | 0.459 |
| **final_best** | Focal | 7e-4 | [0.5, 1, 1, 1.3, 1.3] | **0.491** |

The honest takeaway: I gained **0.6 percentage points** of test macro-F1 over the cross-entropy baseline after a multi-week tuning campaign. The recipe matters, but the architecture mattered far more.

There's also a humbling observation buried in this table. `tuned` and `final_best` share **identical hyperparameters and the same random seed (21)**, yet differ by 3.2 pp on test macro-F1. This is TPU non-determinism: non-associative floating-point reductions across cores, XLA compilation choices that vary between runs, things like that. The variance band is *comparable in size* to the entire ablation range, which means some of the apparent ranking between hyperparameter variants is partially noise. If I were doing this again I would budget seeds the way I budget hyperparameters.

## The augmentation surprise

The single most important hyperparameter in the entire study was data augmentation. Not learning rate, not weight decay, not loss function. Augmentation.

![Hyperparameter ablation summary](/assets/img/blog/tpu/ablation_summary_bar.png)

The "conservative" recipe is:

- random translation up to **6** pixels
- random rotation up to **10°**
- brightness jitter δ = 0.05
- contrast jitter [0.95, 1.05]
- **no horizontal flip** (left/right asymmetry can be diagnostic in mammograms)

The "weak" recipe drops the translation to 4 pixels, the rotation to 8°, and removes brightness/contrast jitter entirely. That's it. That is the full delta.

![Weak vs conservative augmentation training curves](/assets/img/blog/tpu/ablation_augmentation.png)

Weak augmentation collapsed the model from **0.485 to 0.270** test macro-F1, a 21.5 pp drop. The training loss curves tell the story: under weak augmentation, train loss falls faster (overfitting), validation macro-F1 plateaus below 0.25 and then *declines*, triggering early stopping at epoch 11. The model has the capacity to memorise the dataset, and removing two pixels of translation and the brightness jitter is enough to let it.

By comparison, every other single-factor ablation I ran lived inside a 3.5 pp range. Augmentation was an order of magnitude more impactful than every other knob put together.

![LR ablation: validation F1 trajectories](/assets/img/blog/tpu/ablation_lr.png)

## Where the TPU pipeline lands overall

![Per-class F1 for the best models](/assets/img/blog/tpu/dl_per_class_f1_best.png)

The best TPU model, ResNet34 `final_best`, reaches 0.491 test macro-F1 on the 5-class task. That's a meaningful jump over the strongest classical baseline (PCA + HistGradientBoosting at 0.486 balanced accuracy on the *easier* 3-class task). But it is also clearly outclassed by the PyTorch pipeline once pretrained ImageNet weights enter the picture:

![Cross-pipeline comparison: TPU from scratch vs. GPU pretrained](/assets/img/blog/tpu/cross_pipeline_comparison.png)

A Swin-Tiny stabilised on a single L40S GPU with ImageNet pretraining reaches **0.594 test macro-F1**, a +10 pp gap over the best from-scratch TPU model. The gap shows up most strongly on the hardest, smallest classes: `BENIGN_MASS` (+16.1 pp) and `MALIGNANT_CALCIFICATION` (+13.0 pp). That is exactly where transfer learning is supposed to help.

![Confusion matrices: ResNet34 TPU vs. Swin-Tiny GPU](/assets/img/blog/tpu/dl_confusion_best_models.png)

Pretraining wins. Even features learned on natural images transfer remarkably well to grayscale mammographic patches when target data is limited. If clinical performance is the only thing you care about, you absolutely take the pretrained model. The reason to run the TPU pipeline anyway was that I wanted to learn the JAX/Flax/TPU stack from the inside, and the only honest way to learn it is to ship 15 real experiments through it.

## Lessons I would carry into the next project

1. **Augmentation is a load-bearing hyperparameter.** Treat it with the same rigor as the optimizer. A 2-pixel translation difference moved my macro-F1 by 21 percentage points.
2. **Budget seeds, not just configs.** TPU non-determinism gave me a 3.2 pp run-to-run band on identical seeds. Reporting a single number for a single seed is dishonest at this scale; I should have run each "interesting" config 3–5 times.
3. **Log the LR you actually applied.** Reconstructing schedules from configs after the fact works, but only because the schedule is deterministic. I got lucky.
4. **From-scratch on a small medical-imaging dataset is a hard mode you do not need to play.** If the goal is clinical performance, start with pretrained weights and a sensible fine-tuning recipe. The from-scratch numbers are only worth chasing if the *learning* is the point, which, for me, it was.
5. **JAX/Flax on TPU is genuinely lovely once you internalise the functional pattern.** `pmap` makes data parallelism legible. The painful parts are everything *around* training: pretrained weight loading, ONNX export, GradCAM. None of those exist as drop-in tools the way they do in PyTorch. Pick the stack that matches the problem.
6. **Architecture > tuning, on small datasets.** A multi-week ResNet34 tuning campaign moved the needle 0.6 pp. Switching to a pretrained Swin-Tiny moved it +10 pp. Choose your battles.

The full thesis report (with confusion matrices, per-class breakdowns, GradCAM analyses, and the GPU pipeline) is in the [project repo](https://github.com/Davidnet/CM3070-curated_breast_imaging_ddsm). The TPU pipeline lives under [`CM3070-Models-Implementation-Evaluation/models-training-with-tpus/`](https://github.com/Davidnet/CM3070-curated_breast_imaging_ddsm/tree/main/CM3070-Models-Implementation-Evaluation/models-training-with-tpus). Configs, training scripts, and the evaluation harness are all there.

If you have questions or spot something I got wrong, my [contact details](/contact) are on the front page.
