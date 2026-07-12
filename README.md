Car Angle Classifier

A ResNet34-based deep learning model that classifies car images into 8 rotation angles (0°, 40°, 90°, 130°, 180°, 230°, 270°, 320°) — the same category of problem I work with daily while validating AI-driven angle classification pipelines in automotive image cataloguing.

I built this end-to-end, from raw dataset to a trained, evaluated model with GitHub-tracked results, to go beyond validating these systems and actually understand how they're built.

Why this project
Angle classification sounds simple until you look closely at where models actually fail. I wanted to build one myself, break it down with a confusion matrix, and understand the failure modes — not just report an accuracy number.

Tech Stack

- **PyTorch** / **Torchvision** — model and training pipeline
- **ResNet34** (ImageNet-pretrained) — transfer learning backbone
- **scikit-learn** — precision/recall/F1 evaluation, confusion matrix
- **Matplotlib** — results visualization
- **Dataset** — [Car Angle Direction Classification (Kaggle)](https://www.kaggle.com/datasets/fushenggg/car-angle-direction-classification)

Approach

Data pipeline — Images loaded with `ImageFolder`, split into train/val, with augmentation (horizontal flip, small rotation jitter, color jitter) to improve generalization across lighting and orientation variance.

Model — ResNet34 pretrained on ImageNet, backbone frozen, final layer replaced and retrained for 8 angle classes. Freezing the backbone was a deliberate speed/accuracy tradeoff: it trains significantly faster and resists overfitting on a moderate-sized dataset, at the cost of some accuracy versus full fine-tuning.

Training — Adam optimizer, step-decay learning rate scheduler, cross-entropy loss, best-checkpoint saving on validation accuracy.

Evaluation — Full classification report (per-class precision/recall/F1) plus a confusion matrix, used specifically to diagnose *which* angles the model struggles with and *why*.
 Results

- Validation accuracy: 66.67% across 8 classes (random baseline: 12.5%)
- Strongest class: 0° (F1 = 0.83)
- Weakest class: 90° (F1 = 0.53)

The interesting part: reading the confusion matrix

The errors aren't random — they follow a clear geometric pattern. The two largest confusion pairs are **90°↔270°** and **130°↔230°**, which are near-mirror-opposite viewing angles of the car.

That's not a coincidence. A car's side profile is close to symmetric, so distinguishing an exact left-side shot from a right-side shot depends on fine details — badge placement, wheel design, small asymmetries — rather than overall silhouette. Meanwhile, the model *never* confuses structurally distinct angles like 0° (front) and 90° (side). That gap is the real signal: the model has clearly learned strong shape-level features, and its errors are concentrated exactly where shape alone isn't enough to disambiguate — which is a much more specific and fixable problem than "the model just isn't accurate enough."

Next iteration: unfreeze the backbone and fine-tune, or apply targeted augmentation to the confusing angle pairs specifically, rather than blindly scaling the whole dataset.

Why ResNet34 + transfer learning

Residual connections** solve the vanishing gradient problem, which is what makes it possible to train a network this deep reliably in the first place.
ImageNet pretraining** means the model starts with strong general visual features (edges, textures, shapes) already learned — only the angle-specific decision boundary needs to be trained, which needs far less data than training from scratch.
Frozen backbone** was chosen for this dataset size and timeline — the right call here, not a universal one.

Project Structure

├── dataset.py          # data loading, augmentation, train/val split
├── model.py             # ResNet34 architecture setup
├── train.py              # training loop with checkpointing
├── evaluate.py         # classification report + confusion matrix
├── sort_dataset.py    # raw dataset -> train/val folder structure
├── requirements.txt
└── confusion_matrix.png

What I'd do with more time

- Fine-tune the full backbone with a larger dataset to close the gap on the confused angle pairs
- Targeted augmentation for the 90°/270° and 130°/230° classes specifically
- Benchmark against EfficientNet or a ViT-based backbone

Author

Tannu Kumari — [GitHub](https://github.com/TannuRajput77) · [LinkedIn](https://linkedin.com/in/tannu-singh1)
