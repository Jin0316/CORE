# Datasets

This directory holds the data used for training and evaluation. The raw images and
JSON files are **not** tracked in git — this README documents the layout and the JSON
keys the code actually reads, so the data can be reproduced / placed correctly.

Mount this folder at `/workspace/datasets/` (the paths hard-coded in the configs and
builders point there).

```
datasets/
├── safe_eraser/              # Safe-Eraser — concept unlearning (safe_PO tasks)
├── imagenet-r/               # ImageNet-R — image-classification unlearning (safe_PO_IN tasks)
└── Standard_VLM_Benchmarks/  # general vision-language ability (utility) benchmarks
```

---

## 1. `safe_eraser/` — Safe-Eraser

**Used by the pipeline:** `dataset/all_train.json` (training, `safe_PO`) ·
`dataset/all_val.json` (evaluation). Images are relative to `dataset/`.

Each record is one **keyword** (a harmful concept) carrying a harmful and an unharmful
image, plus harmful and benign questions. 

```jsonc
{
  "keyword": "BombAttack",                  // harmful concept — selects a task's data
  "category": "Violence",                   // one of 6 categories — selects the refusal template
  "image_id": "img/Violence/7.jpg",         // HARMFUL image
  "SDImage_path": "pdset/image/Violence/train/40.jpg",   // UNHARMFUL (SD-generated) image
  "unsafe_pairs": [ { "question": "..." } ],             // HARMFUL text — questions to refuse
  "UnharmPair_image1": { "Question": "..." },            // benign question about the image
  "UnharmPair_image2": { "Question": "..." }
}
```

- **Training** (`safe_PO`): for records whose `keyword` is in the current task, each
  `unsafe_pairs[].question` is paired with the harmful `image_id`; the target answer is
  a refusal template chosen by `category` (templates are defined in code:
  `CORE_2026_CVPR/minigpt4/datasets/datasets/prompts.py`).
- **Evaluation** — the four image × text combinations read:

  | mode | image key | question key |
  |------|-----------|--------------|
  | `HH` harmful img + harmful txt     | `image_id`     | `unsafe_pairs[].question` |
  | `HU` harmful img + unharmful txt   | `image_id`     | `UnharmPair_image1/2.Question` |
  | `UH` unharmful img + harmful txt   | `SDImage_path` | `unsafe_pairs[].question` |
  | `UU` unharmful img + unharmful txt | `SDImage_path` | `UnharmPair_image1/2.Question` |

---

## 2. `imagenet-r/` — ImageNet-R (classification unlearning)

**Used by the pipeline:** `imagenet_r_train.json` (training, `safe_PO_IN`) ·
`imagenet_r_test.json` (evaluation). Images are relative to `imagenet-r/imagenet-r/`.

**Keys used:**

```jsonc
{
  "keyword": "goldfish",                            // class name — groups tasks / filters records
  "image_id": "train/n01443537/videogame_5.jpg",    // image path
  "category": "image_classification"
}
```

For `safe_PO_IN` tasks the model is trained to refuse the classification of the
keywords in the current task. (The classification prompt and refusal answer are sampled
from fixed pools in code, so the file's `question` field is not relied on.)

---

## References

- **Safe-Eraser** — *SafeEraser: Enhancing Safety in Multimodal Large Language Models
  through Multimodal Machine Unlearning.* In Findings of the Association for
  Computational Linguistics (ACL), 2025.
- **ImageNet-R** — *The Many Faces of Robustness: A Critical Analysis of
  Out-of-Distribution Generalization.* In Proceedings of the IEEE/CVF International
  Conference on Computer Vision (ICCV), 2021.
