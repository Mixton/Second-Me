# Patch minimal pour ajouter un pipeline d’entraînement LoRA (Qwen3-8B & DeepSeek-R1-Distill-8B)

> Cible du repo : `Mixton/Second-Me` — branche `hotfix/train`

Ce patch ajoute un sous-module d’entraînement basé sur `transformers` + `trl` + `peft` (QLoRA), avec templates Qwen3 et DeepSeek R1-Distill, un Makefile, un Dockerfile CUDA, et des exemples.

---

## Nouvelle arborescence

```
Second-Me/
├─ train/
│  ├─ requirements-train.txt
│  ├─ Dockerfile.backend.cuda.train
│  ├─ Makefile
│  ├─ configs/
│  │  ├─ qwen3-8b.lora.sft.yaml
│  │  └─ deepseek-r1-distill-8b.lora.sft.yaml
│  ├─ data/
│  │  └─ sample.jsonl
│  └─ src/
│     ├─ train_sft.py
│     ├─ data_utils.py
│     ├─ templates.py
│     └─ merge_lora.py
└─ (le reste du repo inchangé)
```

---

## `train/requirements-train.txt`

```txt
transformers>=4.51.0
accelerate>=0.34.2
peft>=0.11.1
trl>=0.10.1
bitsandbytes>=0.43.3
datasets>=2.21.0
evaluate>=0.4.2
scikit-learn>=1.5.1
sentencepiece>=0.2.0
protobuf>=5.27.0
flash-attn>=2.6.3; platform_system=="Linux" and platform_machine=="x86_64"
pyyaml>=6.0.2
```

> **Notes** :
>
> * `flash-attn` nécessite CUDA ≥ 12.1 et un GPU récent ; retirez-le si besoin.
> * Pour A100/H100, laisser activé. Sur RTX 3090/4090 avec CUDA 12.x, ok.

---

## `train/Dockerfile.backend.cuda.train`

```Dockerfile
FROM nvidia/cuda:12.1.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget build-essential python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY train/requirements-train.txt /app/requirements-train.txt
RUN pip install -r /app/requirements-train.txt

# Optionnel : pour servir ensuite via vLLM
RUN pip install vllm>=0.6.2.post1 --extra-index-url https://download.pytorch.org/whl/cu121

COPY train/src /app/src
COPY train/configs /app/configs
COPY train/data /app/data

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

ENTRYPOINT ["bash"]
```

---

## `train/Makefile`

```Makefile
PY=python3
ACC=accelerate
CFG?=configs/qwen3-8b.lora.sft.yaml

.PHONY: env train merge serve-vllm

env:
	pip install -r requirements-train.txt

train:
	$(ACC) launch src/train_sft.py --config $(CFG)

merge:
	$(PY) src/merge_lora.py --config $(CFG)

serve-vllm:
	vllm serve $$(python3 -c 'import yaml,sys;print(yaml.safe_load(open("$(CFG)") ).get("base_model"))') \
	  --tensor-parallel-size 1 --max-model-len 32768 \
	  --adapter "outputs/adapter" || true
```

---

## `train/configs/qwen3-8b.lora.sft.yaml`

```yaml
# SFT LoRA pour Qwen3-8B
base_model: Qwen/Qwen3-8B
output_dir: outputs/qwen3-8b-lora
chat_template: qwen3
use_thinking_mode: true

# Données
train_file: data/sample.jsonl
val_file: null
text_field: text

# Séquences
max_seq_length: 8192
packing: true

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]

# Entraînement
bf16: true
gradient_checkpointing: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
weight_decay: 0.0
num_train_epochs: 2
warmup_ratio: 0.03
logging_steps: 10
save_steps: 500

# Hardware
use_4bit: true # QLoRA
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
ddptype: auto

# Misc
seed: 42
report_to: none
```

---

## `train/configs/deepseek-r1-distill-8b.lora.sft.yaml`

```yaml
# SFT LoRA pour DeepSeek-R1-Distill-Llama-8B
base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
output_dir: outputs/deepseek-r1d-8b-lora
chat_template: deepseek_r1_distill
use_thinking_mode: true  # force <think> ... </think>

train_file: data/sample.jsonl
val_file: null
text_field: text

max_seq_length: 8192
packing: true

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]

bf16: true
gradient_checkpointing: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
weight_decay: 0.0
num_train_epochs: 2
warmup_ratio: 0.03
logging_steps: 10
save_steps: 500

use_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4

seed: 42
report_to: none
prepend_think_token: true
```

---

## `train/data/sample.jsonl`

> Format JSONL simple pour SFT (tu peux remplacer par tes données). Le champ `text` contient l’entrée **ET** la sortie, formatées par le template choisi.

```json
{"text": "<|im_start|>system\nTu es un assistant utile.<|im_end|>\n<|im_start|>user\nExplique la gravité à un enfant de 5 ans.<|im_end|>\n<|im_start|>assistant\n<think>Je vais simplifier l'explication...<\/think>La gravité, c'est comme un aimant géant...<|im_end|>"}
```

---

## `train/src/templates.py`

```python
from dataclasses import dataclass

@dataclass
class ChatExample:
    system: str
    user: str
    assistant: str

# Qwen3: applique le chat template officiel (format <|im_start|> .. <|im_end|>)
def format_qwen3(example: ChatExample, use_thinking_mode: bool = True):
    pre = "<|im_start|>system\n" + (example.system or "") + "<|im_end|>\n"
    pre += "<|im_start|>user\n" + example.user + "<|im_end|>\n"
    ans = example.assistant
    if use_thinking_mode and "<think>" not in ans:
        ans = f"<think>\n{ans.splitlines()[0]}\n</think>\n" + "\n".join(ans.splitlines()[1:])
    pre += "<|im_start|>assistant\n" + ans + "<|im_end|>"
    return pre

# DeepSeek R1 Distill (Llama/Qwen style) avec balises <think>
# On force un préfixe <think> pour stabiliser le raisonnement si demandé.
def format_deepseek_r1_distill(example: ChatExample, use_thinking_mode: bool = True, prepend_think_token: bool = True):
    pre = "<|im_start|>system\n" + (example.system or "") + "<|im_end|>\n"
    pre += "<|im_start|>user\n" + example.user + "<|im_end|>\n"
    ans = example.assistant
    if use_thinking_mode:
        if prepend_think_token and not ans.strip().startswith("<think>"):
            ans = "<think>\n" + ans
        if "</think>" not in ans:
            ans = ans + "\n</think>"
    pre += "<|im_start|>assistant\n" + ans + "<|im_end|>"
    return pre
```

---

## `train/src/data_utils.py`

```python
import json
from datasets import load_dataset
from typing import Optional, Dict

# Charge un dataset JSONL où chaque ligne contient {"text": "..."}
# Si tu veux des champs séparés (system/user/assistant), adapte ici.

def load_text_dataset(path: str, text_field: str = "text"):
    if path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=path, split="train")
        return ds.map(lambda x: {"text": x[text_field]})
    else:
        raise ValueError("Formats supportés: .jsonl")
```

---

## `train/src/train_sft.py`

```python
import os
import yaml
from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_utils import load_text_dataset

@dataclass
class TrainConfig:
    base_model: str
    output_dir: str
    chat_template: str = "qwen3"  # qwen3 | deepseek_r1_distill
    use_thinking_mode: bool = True

    train_file: str = "data/sample.jsonl"
    val_file: Optional[str] = None
    text_field: str = "text"

    max_seq_length: int = 8192
    packing: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    bf16: bool = True
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    num_train_epochs: int = 2
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500

    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    seed: int = 42
    report_to: str = "none"

    # deepseek options
    prepend_think_token: bool = False


def load_cfg(path: str) -> TrainConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def get_bnb_cfg(cfg: TrainConfig):
    if not cfg.use_4bit:
        return None
    compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        llm_int8_threshold=6.0,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    bnb_cfg = get_bnb_cfg(cfg)
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = cfg.target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"
    ]

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)

    # Dataset déjà formatté (voir sample.jsonl). Si besoin de construire depuis system/user/assistant,
    # crée un mapper ici et appelle les helpers de templates.py
    train_ds = load_text_dataset(cfg.train_file, text_field=cfg.text_field)

    sft_conf = SFTConfig(
        output_dir=cfg.output_dir,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=cfg.report_to,
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        args=sft_conf,
    )

    trainer.train()

    # Sauvegarde de l'adapter LoRA uniquement (léger)
    os.makedirs(cfg.output_dir, exist_ok=True)
    adapter_dir = os.path.join(cfg.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to: {adapter_dir}")

if __name__ == "__main__":
    main()
```

---

## `train/src/merge_lora.py`

```python
import os, yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Fusionne l'adapter LoRA dans le modèle base (pour servir sans PEFT ou pour quantifier ailleurs)
# ATTENTION: produit un modèle FP16/BF16 complet, volumineux.

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = cfg["base_model"]
    out = os.path.join(cfg["output_dir"], "merged")
    adapter = os.path.join(cfg["output_dir"], "adapter")

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out)
    print("Merged model saved to:", out)

if __name__ == "__main__":
    main()
```

---

## Utilisation

### 1) En local (sans Docker)

```bash
cd train
make env
make train CFG=configs/qwen3-8b.lora.sft.yaml
# ou
make train CFG=configs/deepseek-r1-distill-8b.lora.sft.yaml
```

### 2) Avec Docker

```bash
cd train
docker build -t secondme-train -f Dockerfile.backend.cuda.train ..
# lance un shell interactif
docker run --gpus all -it --rm -v $(pwd):/app/train -w /app secondme-train
# dedans
make train CFG=configs/qwen3-8b.lora.sft.yaml
```

### 3) Servir pour essais (vLLM + LoRA)

```bash
cd train
make serve-vllm CFG=configs/qwen3-8b.lora.sft.yaml
# puis appeler l'endpoint vLLM habituel
```

### 4) Fusionner l’adapter (optionnel)

```bash
cd train
make merge CFG=configs/qwen3-8b.lora.sft.yaml
# modèle fusionné dans outputs/.../merged
```

---

## Points d’intégration dans Second-Me

* **Inference web actuelle (llama.cpp)** : ne charge pas les adapters PEFT. Deux options :

  1. **Servir via vLLM** (recommandé pour garder LoRA séparé).
  2. **Fusion LoRA → FP16** avec `merge_lora.py`, puis conversion externe vers GGUF si vous souhaitez rester sur llama.cpp (scripts de conversion non inclus ici).
* **Config** : vous pouvez ajouter une variable d’env `ADAPTER_DIR` à votre backend si vous souhaitez charger l’adapter côté serveur (si vous passez par vLLM, utilisez l’option `--adapter`).

---

## Remarques

* Adjustez `max_seq_length`, `packing`, et le batch selon votre VRAM.
* Pour des données multi-champs (system/user/assistant), stockez en JSONL et mappez vers `text` en appliquant les fonctions de `templates.py` avant l’apprentissage.
* DeepSeek R1 « full » n’est pas FT dans `transformers` ; utilisez les distillations (ce patch le fait).

---

**C’est prêt à coller dans `train/` et à lancer.** Si tu me donnes tes contraintes GPU/données, je peux préremplir un config adapté et un mapper system/user/assistant → `text`.

---

## Profil matériel : RTX 5060 (16 GB VRAM) / 32 GB RAM — réglages conseillés

### TL;DR

* **Modèles visés** : Qwen/Qwen3-8B, deepseek-ai/DeepSeek-R1-Distill-Llama-8B
* **Stratégie** : QLoRA 4‑bit (NF4) + gradient checkpointing, **seq\_len 4 096** (monter à 6 144 si ça passe), **batch=1**, **grad\_accum=24–32**.
* **Objectif** : tenir en 16 GB en gardant un contexte confortable et une stabilité d’entraînement.

### Overrides suggérés (Qwen3-8B)

Dans `train/configs/qwen3-8b.lora.sft.yaml`, remplacez :

```yaml
max_seq_length: 4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 24
lora_r: 8           # ↓ VRAM vs 16, impact qualité modéré
lora_alpha: 16
lora_dropout: 0.1
use_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
logging_steps: 10
save_steps: 1000
```

**Astuce stabilité** : si OOM ou lenteurs, passez `max_seq_length: 3072` puis remontez par paliers.

### Overrides suggérés (DeepSeek‑R1‑Distill‑Llama‑8B)

Dans `train/configs/deepseek-r1-distill-8b.lora.sft.yaml` :

```yaml
max_seq_length: 4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 24
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
prepend_think_token: true   # garde le <think> stable
use_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
```

### Fallback attention (si Flash‑Attn indisponible ou OOM)

Dans `train/src/train_sft.py`, changez la ligne `attn_implementation="flash_attention_2"` par :

```python
attn_implementation = "sdpa"  # compatible partout, légèrement plus lent
```

*(Vous pouvez aussi garder un toggle via variable d’environnement `ATTN_IMPL=sdpa` et lire `os.getenv`.)*

### Config `accelerate` (offload CPU utile avec 32 GB RAM)

Créez `~/.cache/huggingface/accelerate/default_config.yaml` :

```yaml
compute_environment: LOCAL_MACHINE
device_placement: true
distributed_type: NO
mixed_precision: bf16
num_processes: 1
offload_optimizer_device: cpu
offload_param_device: none
use_cpu: false
```

> L’offload **optimizer→CPU** économise \~1‑2 GB VRAM en QLoRA 8B.

### Commandes

```bash
# Qwen3-8B
make train CFG=configs/qwen3-8b.lora.sft.yaml

# DeepSeek R1 Distill 8B
make train CFG=configs/deepseek-r1-distill-8b.lora.sft.yaml
```

### Servir l’adapter en 16 GB (vLLM)

```bash
make serve-vllm CFG=configs/qwen3-8b.lora.sft.yaml
# vLLM chargera le base model + l’adapter; utilisez --max-model-len 4096 si besoin
```

### Conseils pratiques 16 GB

* **Datas longues** : privilégiez l’option `packing: true` et filtrez les exemples > `max_seq_length`.
* **Warmup** : `warmup_ratio: 0.03` convient ; si set petite, descendez à `0.01`.
* **Grad norm** : si pertes instables, ajoutez `max_grad_norm: 0.3` dans la config SFT (`SFTConfig`).
* **Montée progressive** : démarrez à `seq_len=3072`, si stable passez à `4096`, puis `5120/6144`.
* **Raisonnement (<think>)** : gardez `use_thinking_mode: true` (Qwen3) et `prepend_think_token: true` (R1‑Distill).

> Avec ces réglages, l’entrainement LoRA 8B **tient dans 16 GB** tout en gardant un contexte 4k. Pour des contextes 8k, réduisez `lora_r` à 4 et augmentez `grad_accum` (débit plus faible).
