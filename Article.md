# Guide pratique (2025) — Quantization LLM sur H100 et alternatives (TRT-LLM, vLLM, GGUF)

## Executive summary (≤12 lignes)

* Objectif: maximiser perf/qualité/coût pour l'inférence LLM sur H100 (et alternatives) via quantization.
* Par défaut H100: **FP8 end‑to‑end (poids+activations) + KV FP8** → débit ↑, latence 1st‑token ↓, VRAM ÷2, qualité ≈ FP16 si calibré.
* Alternative universelle: **INT8 W8A8 (SmoothQuant)** quand FP8 indisponible/instable.
* Densité maximale: **INT4 poids‑seuls (AWQ/GPTQ) + KV FP8** → très compact, légère perte style/cohérence.
* Quand **ne pas** utiliser FP8: couches à grande dynamique non stabilisées, tâches de raisonnement fragiles non calibrées, absence de TE/TC FP8.
* Quand **ne pas** utiliser INT8 W8A8: calibration insuffisante, chemins kernels non int8, CPU pur.
* Quand **ne pas** utiliser INT4: exigences de fidélité élevées (raisonnement long, style strict), absence d’évaluation qualitative.
* Stacks: **TRT‑LLM** = perf absolues H100 ; **vLLM** = OSS flexible haut débit ; **llama.cpp/GGUF** = edge/proto.
* KV: **KV FP8** ≈ ×2 sessions ou ×2 contexte à VRAM constante (voir formule).
* Méthodo: bench réplicables (3 scénarios, 2 tailles de modèles, 3 dtypes + INT4), publier prompts/seed/cmds/metrics.
* Prod: SLOs (p99 1st‑token, tokens/s, VRAM/req, OOM, waste KV), runbook rollback FP8↔INT8↔FP16.
* Licences: quantizer ≠ change de licence ; suivre la plus restrictive des sources ; partager scripts/deltas plutôt que poids si clause restrictive.

---

## Table des matières

* Les bases : tenseurs, poids, activations, KV‑cache
* Formats numériques : FP32, BF16, FP16, FP8, INT8, INT4
* H100 et FP8 : ce qui change
* KV‑cache : FP16 vs FP8 (+ formule GQA)
* Méthodes de quantization clés
* Piles logicielles : TRT‑LLM vs vLLM vs llama.cpp/GGUF
* “8‑bit” sous vLLM : que choisir ?
* Recommandations concrètes (cas 2×H100)
* Pipelines type (déploiement)

  * Pipeline A — TRT‑LLM FP8
  * Pipeline B — TRT‑LLM INT8 (SmoothQuant)
  * Pipeline C — vLLM FP8 / INT8
  * Pipeline D — llama.cpp / GGUF (Q8\_0 / Q4\_K\_\*)
* Tableau décisionnel « quand choisir quoi »
* Bench réplicables minimalistes (méthodo)
* Graphe coût/latence vs qualité (comment le tracer)
* Calibration & garde‑fous FP8/INT8 (checklist)
* Observabilité prod (SLOs)
* Opérations / runbook
* Licences (table simple et actionnable)
* Choisir sa quantization (arbre de décision)
* Commandes types (cheat‑sheet)
* Points de contrôle (qualité)
* TL;DR
* Sources

---

## Les bases : tenseurs, poids, activations, KV‑cache

Un tenseur est un tableau multi‑dimensionnel de nombres (scalaires, vecteurs, matrices, etc.).

Dans un LLM :

* Les **poids** (paramètres) sont stockés sur disque puis chargés en VRAM sous forme de tenseurs.
* Les **activations** sont les résultats intermédiaires calculés pendant l’inférence.
* Le **KV‑cache** contient les Keys et Values de l’attention, conservés au fil de la génération pour accélérer l’auto‑régression.

Sur des serveurs modernes, la gestion du KV‑cache est optimisée (Paged Attention, in‑flight batching). **PagedAttention** stocke les K/V en blocs non contigus pour réduire le gaspillage mémoire.

**Empreinte mémoire à l’inférence** = poids du modèle + activations temporaires + KV‑cache (taille qui croît avec la longueur de contexte). En pratique, les **poids dominent** souvent (≈65% sur un 13B), le **KV‑cache ≈30%** (selon longueur), les activations une part minime.

**Overflow / Underflow (rappel)**

* Overflow : valeur > plage représentable → ∞.
* Underflow : valeur trop petite → arrondie à 0.

Les formats à petite plage (FP8, INT4/INT8) nécessitent **calibration** (ex. SmoothQuant) pour lisser les outliers avant quantization.

---

## Formats numériques : FP32, BF16, FP16, FP8, INT8, INT4

### Résumé des formats

| Format       | Bits | Expo/Mantisse | Plage dynamique (≈)     | Utilisation & remarques                                                   |
| ------------ | ---: | ------------: | ----------------------- | ------------------------------------------------------------------------- |
| FP32         |   32 |          8/23 | \~1e‑38 → 1e+38         | Référence entraînement ; coûteux.                                         |
| BF16         |   16 |           8/7 | \~1e‑38 → 1e+38 (≈FP32) | Mixed‑precision ; même range FP32, mantisse plus courte.                  |
| FP16         |   16 |          5/10 | \~1e‑4 → 6.5e+4         | Standard inférence GPU (Tensor Cores).                                    |
| FP8 E4M3     |    8 |           4/3 | \~1e‑2 → \~4.5e+2       | Inférence H100 ; calibration impérative.                                  |
| FP8 E5M2     |    8 |           5/2 | \~1e‑2 → \~5.7e+4 (+∞)  | Plutôt pour grads/backward ou KV à large dynamique.                       |
| INT8 (W8A8)  |    8 |        entier | via échelles            | Poids+activations 8‑bit ; SmoothQuant pour stabilité ; Tensor Cores int8. |
| INT4 (poids) |    4 |        entier | via échelles/groupes    | Compression ×4 vs FP16 ; AWQ/GPTQ ; légère perte style/cohérence.         |

**Pourquoi préciser “poids + activations” ?**
Beaucoup de quantizations ne touchent que les **poids** (taille disque/VRAM). Si les **activations** restent en FP16, le calcul ne s’accélère pas autant. → **W8A8** (ou **FP8**) tire parti de la quantization sur tout le chemin (GEMM, etc.). Quantifier les activations est délicat (distributions variables) → **calibration** cruciale.

---

## H100 et FP8 : ce qui change

La génération **Hopper (H100)** apporte des Tensor Cores FP8 + **Transformer Engine (TE)** qui gère automatiquement FP16↔FP8 selon un recipe optimal.

**Bénéfices majeurs** : débit ↑ (jusqu’à ×4–5 vs A100 FP16), latence 1er token ↓, mémoire ÷2 vs FP16, **qualité \~FP16** si calibré.

**TRT‑LLM** exploite FP8 natif, in‑flight batching, paged KV‑cache → débits inédits sur H100. En prod, certains gardent un profil FP16 de contrôle au début ; dans la pratique, **FP8 bien calibré est quasi indiscernable**.

---

## KV‑cache : FP16 vs FP8 (+ formule GQA)

* Par défaut, KV en **FP16** = fidélité max mais VRAM élevée.
* Passer le KV en **FP8** ÷2 l’empreinte → plus de contexte et/ou plus de sessions.
* Impact qualité très faible si calibration correcte.
* Le **dtype du KV est indépendant** de celui des poids/activations (ex. modèle FP16 + KV FP8 = quick win VRAM ; modèle FP8 + KV FP16 = précision max attentions).
* **PagedAttention** améliore l’usage KV quelle que soit la précision (pages petites, anti‑fragmentation).

**Formule générique (avec GQA/MQA)**
$\text{KV\_bytes} \approx \text{batch} \times \text{seq\_len} \times L \times \big(2 \times \frac{d_\text{model}}{g}\big) \times \text{bytes\_per\_dtype}$
avec $L$=nb de couches, $d_\text{model}=n_\text{heads}\times\text{head\_dim}$, $g=n_\text{heads}/n_\text{kv\_heads}$ (GQA/MQA).

**bytes\_per\_dtype** (rappel)

| dtype | bytes |
| ----- | ----: |
| fp16  |     2 |
| fp8   |     1 |
| int8  |     1 |

**Conclusion rapide** : **KV FP8** ≈ **×2 sessions** ou **×2 contexte** à VRAM constante.

---

## Méthodes de quantization clés

* **SmoothQuant** (INT8 W8A8, post‑training) : lisse les outliers d’activation via rescaling par couche → W8A8 stable sans fine‑tuning. Modèles >500B avec perte négligeable. Gain mémoire ×2, accélération \~1.5×.
* **AWQ** (INT4 poids‑seuls, Activation‑aware) : protège \~1% de canaux critiques (8/16‑bit), reste en 4‑bit. Qualité SOTA en 4‑bit, y compris instruction‑tuned/multi‑modal.
* **GPTQ** (INT3/INT4 poids‑seuls) : one‑shot avec info de second ordre pour minimiser l’erreur. Très rapide, excellente qualité en 3–4 bits sur grands modèles.
* **LLM.int8 (bitsandbytes)** (poids‑seuls 8‑bit) : vectorisation par colonne + outliers en 16‑bit → \~99.9% des ops en int8. Allège la mémoire ; accélération < W8A8/FP8 sur H100.

---

## Piles logicielles : TRT‑LLM vs vLLM vs llama.cpp/GGUF

**TensorRT‑LLM (NVIDIA)**
Compilateur/runtime qui génère un engine optimisé par GPU.

* **Forces** : FP8 natif H100, INT8 (SmoothQuant), INT4 (AWQ), in‑flight batching, paged KV, multi‑GPU. Perf absolues sur H100 au top.
* **Limites** : spécifique NVIDIA, nécessite build ; engines non portables ; certains modèles exotiques demandent un parser à jour.

**vLLM (open‑source)**
Serveur haut débit avec **PagedAttention** (≈<4% mémoire KV gaspillée).

* **Forces** : FP8, INT8 (W8A8), chargement AWQ/GPTQ 4‑bit, batching dynamique, streaming, Python‑friendly.
* **Limites** : légèrement sous TRT‑LLM (modèle égal), dépend des kernels disponibles.

**llama.cpp / GGUF (CPU & hétérogène)**

* **Forces** : multiplateforme (CPU, Apple Silicon, petits GPU), déploiement simple (.gguf). Parfait pour prototypage/edge.
* **Limites** : loin des perfs H100 ; sur H100, préférer TRT‑LLM/vLLM.

> Compatibilité : engines TRT‑LLM / modèles vLLM **≠** GGUF. Conserver le checkpoint HF original (les builds sont des dérivés).

---

## “8‑bit” sous vLLM : que choisir ?

**Par défaut H100** : `--quantization fp8` + `--kv-cache-dtype fp8`.

```bash
vllm serve $MODEL_ID \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
```

**Alternative INT8 (W8A8) via SmoothQuant** :

```bash
vllm serve $MODEL_ID \
  --quantization int8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
```

> À ne pas confondre avec `load_in_8bit=True` (bitsandbytes, poids‑seuls).

---

## Recommandations concrètes (cas 2×H100)

* **FP8 end‑to‑end** (poids+acts FP8, KV FP8) = sweet spot H100 : qualité \~FP16, latence ↓, VRAM ↓.
* **INT8 SmoothQuant (W8A8)** : stable, rapide, universel (légèrement moins rapide que FP8).
* **INT4 AWQ (+ KV FP8)** : densité max (ex. 70B ≈ \~40 Go) ; légère dégradation style/cohérence, à valider sur prompts sensibles.
* **Validation** : calibrage + jeu de prompts réaliste, A/B blindé (FP16 vs quant), PPL, distinct‑n ; ajuster decoding (repetition\_penalty, temperature/top\_p) si besoin.

---

## Pipelines type (déploiement)

### Pipeline A — TRT‑LLM FP8 (recommandé H100)

**Exporter HF → TRT‑LLM checkpoint**

```bash
python3 examples/llama/convert_checkpoint.py \
  --model_dir /models/YourModelHF \
  --output_dir /out/trtllm_ckpt \
  --dtype float16 --tp_size 2
```

**Builder l’engine FP8 + KV FP8**

```bash
trtllm-build \
  --checkpoint_dir /out/trtllm_ckpt \
  --output_dir /out/engine_fp8_tp2 \
  --tp_size 2 --max_batch_size 16 \
  --max_input_len 16384 --max_output_len 1024 \
  --use_fp8 --use_fp8_kv_cache
```

**Servir**

```bash
trtllm-serve --engine_dir /out/engine_fp8_tp2 --port 8000
```

### Pipeline B — TRT‑LLM INT8 (SmoothQuant)

* Export HF → TRT‑LLM (idem).
* Calibration SmoothQuant (dataset court).
* **Build INT8 (KV en FP8 recommandé)**

```bash
trtllm-build ... --quantize int8 --use_fp8_kv_cache ...
```

* Serve via `trtllm-serve`.

### Pipeline C — vLLM FP8 / INT8

**FP8**

```bash
vllm serve mistralai/Mistral-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192
```

**INT8 (W8A8)**

```bash
vllm serve ORG/MODELE-HF \
  --quantization int8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
```

### Pipeline D — llama.cpp / GGUF (Q8\_0 / Q4\_K\_\*)

**Conversion & quantization**

```bash
# Convertir HF -> GGUF FP16
python convert-hf-to-gguf.py NomDuModeleHF --outfile modele.gguf

# Quantifier en 4 bits
./quantize modele.gguf modele-q4_0.gguf q4_0
```

---

## Tableau décisionnel « quand choisir quoi »

| Contrainte / Option       |                  FP8 (H100)                 |                  INT8 (W8A8)                 |          INT4 (AWQ/GPTQ)          | GGUF (llama.cpp) |
| ------------------------- | :-----------------------------------------: | :------------------------------------------: | :-------------------------------: | :--------------: |
| Latence minimale          |                      ✅                      |                       ✅                      |                 ⚠️                |         ❌        |
| Débit élevé (multi‑users) |                      ✅                      |                       ✅                      |                 ⚠️                |         ❌        |
| VRAM très limitée         |                      ⚠️                     |                       ✅                      |                 ✅                 |         ✅        |
| Qualité ≈ FP16 requise    |                      ✅                      |                      ⚠️                      |                 ⚠️                |        ⚠️        |
| Portabilité (CPU/edge)    |                      ❌                      |                      ⚠️                      |                 ⚠️                |         ✅        |
| Simplicité d’intégration  |                      ⚠️                     |                       ✅                      |                 ⚠️                |         ✅        |
| Flags vLLM typiques       | `--quantization fp8` `--kv-cache-dtype fp8` | `--quantization int8` `--kv-cache-dtype fp8` |          charger AWQ/GPTQ         |  `.gguf` presets |
| TRT‑LLM typiques          |        `--use_fp8 --use_fp8_kv_cache`       |     `--quantize int8 --use_fp8_kv_cache`     | `--quantize int4` (selon support) |        n/a       |

> Légende: ✅ recommandé • ⚠️ dépend du use‑case • ❌ non adapté.

---

## Bench réplicables minimalistes (méthodo)

**Scénarios**

1. **Faible latence** (batch=1, prompt court → 1st‑token & tokens/s).
2. **Charge mixte** (batch dynamique 1–32, prompts hétérogènes).
3. **Long contexte** (≥16k), avec KV FP8 vs FP16.

**Modèles** : un **7B** et un **\~70B**.

**Dtypes** : **FP16**, **FP8**, **INT8 (W8A8)** + **INT4 poids‑seuls**.

**Publier** : prompt set, seed, exact **cmds** (serve & client), commits/versions, hardware (2×H100‑80GB, NVLink/NVSwitch), paramètres decoding.

**Métriques** :

* **tok/s** (steady‑state), **p50/p99 1st‑token**, **latence totale**.
* **VRAM max** par GPU, **OOM rate**, **waste KV** (fragmentation).
* **Qualité** : PPL (proxy), distinct‑n, taux de refus/hallucinations.

**Exigences** : 3 runs/point, seed fixé, warmup mesuré, cold‑start vs warm.

---

## Graphe coût/latence vs qualité (comment le tracer)

* Axe X: coût/latence (p50/p99 1st‑token, tok/s inversé) ; Axe Y: qualité (PPL↓, proxy de factualité).
* Points: {dtype, stack, modèle}.
* Ajouter bandes d’erreur (icône p50/p99) et iso‑courbes de VRAM.
* Interprétation: zones pareto‑optimales (FP8 e2e souvent dominante sur H100).

---

## Calibration & garde‑fous FP8/INT8 (checklist)

* **Jeu de calibration**: 512–2k prompts proches prod (multi‑tâches/langues).
* **Exclure** si instable: embeddings, `lm_head`, éventuels blocks spécifiques.
* **Recette**: mix **E4M3/E5M2** par bloc + scaling **per‑tensor/per‑channel**.
* **Sanity checks**: saturation rate, overflow count, cosine sim.
* **Fallback**: `--kv-cache-dtype fp16` pour cas de raisonnement fragile/long contexte.
* **Ajustements decoding**: `repetition_penalty↑ (1.10→1.15)`, `temperature/top_p`.

---

## Observabilité prod (SLOs)

* **Suivre**: p99 1st‑token, tokens/s, VRAM/req, waste KV, OOMs, taux d’abandon, drift qualité (PPL batchée hebdo).
* **Alertes**: p99 > seuil, VRAM > 95%, OOM > 0.5% req, waste KV > 8%.
* **Traces**: files d’attente, in‑flight batching, fragmentation KV, temps kernels.

---

## Opérations / runbook

* **Roll‑forward/back**: FP8 ↔ INT8 ↔ FP16, canary A/B (5% trafic) avant bascule.
* **Compat matrix**: versions pin de drivers/CUDA/cuBLAS/cuDNN/TRT‑LLM/vLLM.
* **Warmup**: profils par modèle, longueur de prompt, batch, sampling.
* **Capacity planning**: dimensionner KV (formule), headroom VRAM, fragmentation cible.

---

## Licences (table simple et actionnable)

> La quantization **ne change pas** la licence. Les merges héritent de la **plus restrictive** des sources. Vérifier la licence **auprès de l’éditeur** avant redistribution.

| Modèle (exemples) | Licence         | Usage commercial         | Redistribution poids **quantifiés** | Stratégie de partage conseillée        |
| ----------------- | --------------- | ------------------------ | ----------------------------------- | -------------------------------------- |
| Famille LLaMA\*   | Licence éditeur | Souvent oui (conditions) | Souvent restreinte                  | Publier **scripts/deltas/LoRA**        |
| Mistral\*         | Licence éditeur | Souvent oui              | Selon clause                        | Scripts/deltas ; éviter poids si doute |
| Qwen\*            | Licence éditeur | Souvent oui              | Selon clause                        | Scripts/deltas ; vérifier exceptions   |
| Modèles NC/MRL    | NC/MRL          | **Non**                  | **Interdit**                        | Aucun poids ; docs & deltas si permis  |

\*Toujours lire la licence spécifique à la version.

---

## Choisir sa quantization (arbre de décision)

```
Besoin ≈ FP16 + perf max sur H100 ? → FP8 e2e (KV FP8)
  └─ FP8 instable / non dispo ? → INT8 W8A8 (KV FP8)
       └─ VRAM ultra‑contrainte / edge ? → INT4 AWQ/GPTQ (KV FP8)
            └─ Portabilité CPU/edge prioritaire ? → GGUF (llama.cpp)
```

---

## Commandes types (cheat‑sheet)

**TRT‑LLM (H100, FP8)**

```bash
python examples/llama/convert_checkpoint.py \
  --model_dir /chemin/vers/modele_hf \
  --output_dir /chemin/vers/output_trtllm_ckpt \
  --dtype float16 --tp_size 2

trtllm-build \
  --checkpoint_dir /chemin/vers/output_trtllm_ckpt \
  --output_dir /chemin/vers/engine_fp8 \
  --use_fp8 --use_fp8_kv_cache \
  --max_batch_size 8 \
  --max_input_len 8192 --max_output_len 1024 \
  --tp_size 2

trtllm-serve --engine_dir /chemin/vers/engine_fp8 --port 8080
```

**vLLM FP8 (W8A8 + KV FP8)**

```bash
vllm serve ORG/MODELE-HF \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
```

**vLLM INT8 (W8A8 + KV FP8)**

```bash
vllm serve ORG/MODELE-HF \
  --quantization int8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
```

**GGUF (llama.cpp)**

```bash
python convert-hf-to-gguf.py NomDuModeleHF --outfile modele.gguf
./quantize modele.gguf modele-q4_0.gguf q4_0
```

---

## Points de contrôle (qualité)

* Jeu de validation réaliste (10–50 prompts, multi‑tours, FR/EN).
* Métriques : PPL, distinct‑n, taux de refus/hallucinations.
* A/B humain à l’aveugle (FP16 vs quant).
* Tuning decoding : repetition\_penalty ↑ (1.10→1.15), ajuster temperature/top\_p.
* Long contexte (≥16k) : tester KV FP8 ou FP16 selon VRAM/discernement.

---

## TL;DR

* **H100 = FP8 natif** → perf ×3–5, latence ↓, VRAM ÷2, qualité \~FP16 si calibré.
* **TRT‑LLM** = perf absolues (FP8/INT8, in‑flight batching, paged KV).
* **vLLM** = OSS très performant (PagedAttention, FP8/INT8, flexible).
* **Choix quant** : FP8 si possible, sinon INT8 (SmoothQuant) ; INT4 (AWQ/GPTQ) pour densité.
* **GGUF (llama.cpp)** = prototypage/edge, pas pour perf H100.
* **Licences** : quantization ≠ change de licence ; respecter NC/MRL ; publier scripts/deltas, pas les poids si restrictions.

---

## Sources

* docs.nvidia.com
* developer.nvidia.com
* nvidia.github.io
* developers.redhat.com
* blog.vllm.ai
* runpod.io
* arxiv.org
* ar5iv.labs.arxiv.org
* medium.com
* qwen.readthedocs.io
* huggingface.co
