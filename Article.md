# Guide Pratique & Stratégique (2025) : Dimensionnement, Déploiement et Quantization de LLM

**Statut :** Validé  
**Cible :** Ingénieurs MLOps, AI Architects, DevOps  
**Dernière mise à jour :** 2026-02-22  

Ce document synthétise des standards et bonnes pratiques pour déployer des LLMs en production, avec un focus sur :
- **Dimensionnement VRAM** (poids + KV-cache + overhead runtime)
- **Quantization** (FP8 / INT8 / INT4) et impacts qualité/perf
- **Moteurs d’inférence** (TensorRT-LLM, vLLM, llama.cpp/GGUF)
- **Décisions d’architecture** (latence, throughput, coûts, contraintes matérielles)

> Note GitHub : le LaTeX inline peut casser sur certains `_` (ex: dans `\text{...}` / `\texttt{...}`) et produire `_' allowed only in math mode`.  
> Les formules critiques sont donc données en **code** pour un rendu 100% fiable.

---

## Table des matières
- [0. Notations et hypothèses](#0-notations-et-hypothèses)
- [1. Fondations : mémoire d’un LLM en inférence](#1-fondations--mémoire-dun-llm-en-inférence)
- [2. Dimensionnement VRAM : combien gagne-t-on ?](#2-dimensionnement-vram--combien-gagne-t-on-)
- [3. Méthodes de quantization clés](#3-méthodes-de-quantization-clés)
- [4. Comparatif des moteurs d’inférence](#4-comparatif-des-moteurs-dinférence)
- [5. Arbre de décision : quelle stratégie déployer ?](#5-arbre-de-décision--quelle-stratégie-déployer-)
- [6. Production : observabilité, SLIs, licences](#6-production--observabilité-slis-licences)
- [7. Cheat-sheet : commandes de déploiement (avec explications)](#7-cheat-sheet--commandes-de-déploiement-avec-explications)
- [8. Références](#8-références)
- [Annexes : mini-calculateur VRAM/KV-cache](#annexes--mini-calculateur-vramkv-cache)

---

## 0. Notations et hypothèses

### Unités
- **GiB** (gibibyte) = `1024^3` bytes (plus précis que “GB”)
- Les “Go” affichés par les GPU vendors sont parfois en base 10 → garde une marge.

### Notations
- `B` = batch_size (requêtes simultanées effectives)
- `S` = sequence_length (tokens contexte total pris en charge : prompt + génération)
- `L` = num_layers
- `H` = hidden_size
- `num_heads` = nombre de têtes attention (Query heads)
- `num_kv_heads` = nombre de têtes K/V (MQA/GQA)
- `gqa_factor = num_heads / num_kv_heads`
- `bytes_kv` = bytes par élément du KV-cache (ex: FP16=2, FP8=1)

---

## 1. Fondations : mémoire d’un LLM en inférence

En production, la VRAM est principalement consommée par trois blocs :

1. **Poids (Weights)**  
   Paramètres du modèle (ce qui “scale” le plus avec le nombre de paramètres).
2. **KV-Cache (Key/Value Cache)**  
   Stockage des clés/valeurs d’attention pour éviter de recalculer l’historique (auto-régression).  
   👉 Croît ~linéairement avec **S** et **B**.
3. **Overhead runtime** (souvent sous-estimé)  
   CUDA context, buffers, allocations du moteur, fragmentation, graph capture, etc.

> Les **activations** en pur “decode” sont souvent moins dominantes que le KV-cache, mais la phase **prefill** peut générer des pics selon le moteur et la config.

### 1.1 Formule pratique du KV-cache (GQA/MQA inclus)

Formule “lisible” (approximation utile pour capacity planning) :

`KV_bytes ≈ B × S × L × 2 × (H / gqa_factor) × bytes_kv`

- `2` = K et V  
- `(H / gqa_factor)` est équivalent à `num_kv_heads × head_dim`

Version “explicite” :

`KV_bytes ≈ B × S × L × 2 × (num_kv_heads × head_dim) × bytes_kv`

#### Exemple (ordre de grandeur)
Sur un 70B typique (GQA), passer le KV-cache de FP16 → FP8 divise **≈ par 2** l’empreinte KV-cache, ce qui permet soit :
- plus de contexte (`S`),
- plus de concurrence (`B`),
- moins de risques d’OOM.

---

### 1.2 PagedAttention : pourquoi ça change tout

**PagedAttention** (vLLM) découpe le KV-cache en blocs “paginés”, ce qui réduit les pertes dues à la réservation + fragmentation.  
Dans le papier vLLM, la **KV cache usage** mesurée monte jusqu’à ~**96.3%** pour vLLM, tandis que des baselines (Orca variants) restent beaucoup plus bas selon les scénarios (réservation/fragmentation).  
👉 Résultat : plus de requêtes “in-flight” à VRAM égale, et meilleure tenue sous charge.

---

## 2. Dimensionnement VRAM : combien gagne-t-on ?

### 2.1 Règle d’or (poids)
Approximation poids seuls :

`Weights_bytes ≈ num_params × bytes_per_weight`

- FP16/BF16 : 2 bytes
- FP8/INT8 : 1 byte (en pratique : +scales/metadata selon méthode)
- INT4 : 0.5 byte (en pratique : +scales/packing + parfois “outliers” en FP16)

> Toujours ajouter une **marge** (souvent 10–25%) pour l’overhead moteur + KV-cache selon ton S et B.

---

### 2.2 Étude de cas (valeurs indicatives)

Hypothèses pour “VRAM totale” :
- KV-cache activé (taille dépendante de `S` et `B`)
- Overhead moteur inclus en “marge”
- Les chiffres restent des **ordres de grandeur** : la vérité dépend de `max_model_len`, batching, backend attention, kernels, etc.

| Modèle cible | Format | VRAM Poids seuls (≈) | VRAM Totale (indicatif) | GPU minimum “confort” | Gain vs FP16 |
|---|---:|---:|---:|---|---:|
| 8B | FP16 | ~16 GiB | ~18–22 GiB | RTX 3090/4090 (24GB) / A10 | réf |
| 8B | FP8 / INT8 | ~8 GiB | ~10–14 GiB | 16GB (selon contexte) | ~-50% |
| 8B | INT4 (AWQ/GGUF) | ~4 GiB | ~6–10 GiB | 12GB possible | ~-75% |
| 70B | FP16 | ~140 GiB | ~150–180 GiB | 2×80GB (TP) | réf |
| 70B | FP8 / INT8 | ~70 GiB | ~75–110 GiB | 1×80GB **si** contexte/batch maîtrisés | ~-50% |
| 70B | INT4 (AWQ/GGUF) | ~35 GiB | ~40–70 GiB | 48GB ou 2×24GB (TP) | ~-75% |

---

### 2.3 Le compromis : VRAM vs Vitesse vs Qualité

#### FP8 / INT8 : “sweet spot” production
- **Mémoire** : ~2× moins de VRAM poids (théorique)
- **Perf** : sur matériel supporté, vLLM indique jusqu’à **~1.6×** de throughput avec impact minimal sur l’accuracy selon modèles/tâches.  
- **KV-cache FP8** : gros levier sur contexte et concurrence.

#### INT4 : “densité / coût”
- **Mémoire** : énorme compression
- **Qualité** : dépend du modèle + méthode (AWQ souvent très bon ratio, GPTQ variable)
- **Perf** : parfois limitée par kernels / déquant / mémoire plutôt que compute pur

---

## 3. Méthodes de quantization clés

Trois familles PTQ dominent l’écosystème :

- **SmoothQuant (INT8 W8A8)**  
  Réduit l’effet des outliers en redistribuant l’amplitude activations↔poids via rescaling.  
  Bon compromis quand tu veux INT8 stable sans retraining.

- **AWQ (INT4)**  
  “Protection” d’une petite partie des poids les plus sensibles, quantization du reste en 4-bit.  
  Très populaire pour servir des modèles lourds avec faible VRAM.

- **GPTQ (INT3/INT4)**  
  One-shot par blocs (approx Hessienne) pour compenser l’erreur de quantization.  
  Très utilisé côté open-source.

---

## 4. Comparatif des moteurs d’inférence

### A) TensorRT-LLM (NVIDIA)
Runtime/stack ultra-optimisé GPU NVIDIA (latence et débit max possibles).
- **Points forts** : FP8, KV-cache FP8, kernels optimisés, serveur compatible OpenAI, endpoint metrics.
- **Tradeoffs** : dépendance CUDA/NVIDIA, phase build/engine.

### B) vLLM (Open-Source)
Standard industriel “serving” pour LLM (PagedAttention + gros throughput sous charge).
- **Points forts** : PagedAttention, config riche KV-cache, quantization FP8/INT8/INT4, API OpenAI-compatible.
- **Tradeoffs** : TRT-LLM peut garder l’avantage sur certains profils latence extrême.

### C) llama.cpp / GGUF (CPU & Edge)
Exécution locale / edge, très pratique pour environnements contraints.
- **Points forts** : portable, large palette de quantizations (Q4_K_M, Q8_0, etc.)
- **Tradeoffs** : pas fait pour exploiter un datacenter GPU au maximum.

---

## 5. Arbre de décision : quelle stratégie déployer ?

1. **H100 / Hopper : perf + qualité**
   - **Poids** : FP8 (W8A8) si possible
   - **KV-cache** : FP8
   - **Moteur** : TensorRT-LLM ou vLLM
   - **Quand** : prod exigeante, gros trafic, contexte long

2. **A100 / Ampere & parc hétérogène : robustesse**
   - **Poids** : INT8 (SmoothQuant) souvent safe
   - **KV-cache** : FP8 si supporté, sinon BF16/FP16
   - **Moteur** : vLLM
   - **Quand** : large compat, déploiements rapides, bon ratio perf/coût

3. **Budget / haute densité**
   - **Poids** : INT4 (AWQ / GGUF)
   - **KV-cache** : FP8 si possible
   - **Moteur** : vLLM (serve) ou llama.cpp (local/edge)
   - **Quand** : chatbots internes, workloads tolérants à légère baisse qualité

---

## 6. Production : observabilité, SLIs, licences

### 6.1 Monitoring (exemples de SLIs)
| Métrique | Seuil d’alerte (exemple) | Actions typiques |
|---|---:|---|
| **TTFT** (time-to-first-token) p99 | > 500 ms | scale-out, profiling kernels, réduire `max_model_len` |
| **Tokens/s** (débit) | chute > 20% | vérifier batching, contention GPU, throttling |
| **OOM rate** | > 0.5% req | réduire `max_model_len`, baisser `B`, KV-cache FP8 |
| **KV-cache waste / residency** | dérive | revoir config cache, prefix caching, warmup |

### 6.2 Licensing (rappel)
La quantization est une transformation technique : **elle ne change pas la licence** du modèle.
- Un modèle fusionné/mergé hérite des contraintes amont.
- Si usage commercial interdit sur le modèle base → interdit aussi sur versions quantized.
- En cas de blocage : partager les **scripts** de repro plutôt que les poids.

---

## 7. Cheat-sheet : commandes de déploiement (avec explications)

### 7.1 vLLM : servir un modèle avec FP8 + KV-cache FP8

**Pourquoi** : rapide à mettre en prod, très bon throughput sous charge, configuration KV-cache riche.

<details>
<summary><strong>Commande vLLM (FP8 + KV FP8)</strong></summary>

```bash
vllm serve mistralai/Mistral-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384

Notes importantes

--kv-cache-dtype fp8 réduit fortement la mémoire KV (utile pour long contexte / plus de concurrence).

Si ton checkpoint ne contient pas d’échelles KV adaptées, tu peux explorer :

--calculate-kv-scales (warmup/calibration)

ou calibration dataset (recommandé en prod).

</details>
7.2 TensorRT-LLM : quantize → build engine → serve (OpenAI-compatible)

Pourquoi : latence et débit max sur GPU NVIDIA, très pertinent sur H100/H200.

<details> <summary><strong>Étape A — (Optionnel) Quantization Toolkit → checkpoint TensorRT-LLM</strong></summary>
# Exemple FP8 + KV-cache FP8 (calibration requise)
python examples/quantization/quantize.py \
  --model_dir $MODEL_HF_DIR \
  --qformat fp8 \
  --kv_cache_dtype fp8 \
  --output_dir $TRTLLM_CKPT_DIR

Le checkpoint exporté peut ensuite être utilisé directement par trtllm-build.

</details> <details> <summary><strong>Étape B — Build de l’engine</strong></summary>
trtllm-build \
  --checkpoint_dir $TRTLLM_CKPT_DIR \
  --output_dir $TRTLLM_ENGINE_DIR
</details> <details> <summary><strong>Étape C — Serving OpenAI-compatible</strong></summary>
trtllm-serve $TRTLLM_ENGINE_DIR --port 8080

Endpoints : /v1/chat/completions, /v1/completions, etc.

Observabilité : /metrics, /health, /version.

</details>
8. Références
Papers

SmoothQuant — Xiao et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML.
https://arxiv.org/abs/2211.10438

AWQ — Lin et al. (2024). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys.
https://arxiv.org/abs/2306.00978

GPTQ — Frantar et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR.
https://arxiv.org/abs/2210.17323

LLM.int8() — Dettmers et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS.
https://arxiv.org/abs/2208.07339

PagedAttention / vLLM — Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention.
https://arxiv.org/abs/2309.06180

Docs / vendors

vLLM FP8 : https://docs.vllm.ai/en/stable/features/quantization/fp8/

vLLM KV-cache FP8 : https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/

TensorRT-LLM docs (build/serve) : https://nvidia.github.io/TensorRT-LLM/

Annexes : mini-calculateur VRAM/KV-cache
def gib(x_bytes: float) -> float:
    return x_bytes / (1024**3)

def weights_gib(num_params: float, bytes_per_weight: float) -> float:
    return gib(num_params * bytes_per_weight)

def kv_cache_gib(B: int, S: int, L: int, H: int, gqa_factor: int, bytes_kv: float) -> float:
    # KV_bytes ≈ B × S × L × 2 × (H / gqa_factor) × bytes_kv
    return gib(B * S * L * 2 * (H / gqa_factor) * bytes_kv)

# Exemple quick check :
# 70B FP16 poids seuls ≈ 70e9 * 2 bytes
print("Weights 70B FP16 ~", weights_gib(70e9, 2), "GiB")

Points “sources vérifiées” que j’ai alignés explicitement :
- vLLM FP8 : **2× réduction mémoire** et **jusqu’à ~1.6× throughput** :contentReference[oaicite:2]{index=2}  
- vLLM KV-cache FP8 + options/calibration :contentReference[oaicite:3]{index=3}  
- PagedAttention : amélioration d’utilisation mémoire KV-cache (paper vLLM) :contentReference[oaicite:4]{index=4}  
- TensorRT-LLM : serveur OpenAI-compatible + `/metrics` :contentReference[oaicite:5]{index=5}  
- TensorRT-LLM : H100 vs A100 (jusqu’à 4.6× max throughput en FP8 selon leur blog) :contentReference[oaicite:6]{index=6}  
- Problème GitHub / underscores en LaTeX :contentReference[oaicite:7]{index=7}  

Si tu veux, je peux aussi te proposer une **section “Bench protocole”** (comment mesurer TTFT/tok/s proprement + profils latency/throughput) calibrée pour H100/A100, mais là tu as déjà une version GitHub “nickel” et sourcée.
::contentReference[oaicite:8]{index=8}
