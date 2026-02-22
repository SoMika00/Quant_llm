# Guide Pratique & Stratégique (2025) : Dimensionnement, Déploiement et Quantization de LLM

**Statut :** Validé | **Cible :** Ingénieurs MLOps, AI Architects, DevOps

Ce document documente les standards architecturaux et les meilleures pratiques pour le déploiement de Large Language Models (LLMs) en environnement de production. Il se concentre sur l'optimisation de l'inférence via la quantization, le dimensionnement de la VRAM sur GPU (NVIDIA H100, A100, RTX), et le choix des moteurs d'inférence (TRT-LLM, vLLM).

---

## 1. Fondations : Gestion de la Mémoire d'un LLM

Pour maîtriser la quantization, il est impératif de comprendre la répartition de l'empreinte mémoire d'un modèle lors de l'inférence. L'occupation de la VRAM se divise en trois piliers :

* **Poids (Weights) :** Les paramètres du modèle appris durant l'entraînement. Ils dictent la taille minimale sur disque et en mémoire.
* **Activations :** Les résultats intermédiaires calculés à chaque étape de l'inférence. Leur empreinte est généralement minime.
* **KV-Cache (Key-Value Cache) :** Tenseurs de clés et valeurs du mécanisme d'attention, conservés au fil de la génération pour éviter de recalculer l'historique (auto-régression). Il croît de manière linéaire avec la longueur du contexte et le nombre de requêtes simultanées.

### La Formule du KV-Cache (avec GQA)
L'estimation de la mémoire allouée au KV-cache se calcule selon la formule suivante :

`KV_bytes ≈ batch_size × sequence_length × num_layers × (2 × hidden_size) × (bytes_per_dtype / gqa_factor)`

### L'optimisation PagedAttention
Indépendamment du format numérique, des algorithmes comme **PagedAttention** (introduit par vLLM) optimisent l'usage du KV-cache en le découpant en pages non contiguës. Cela réduit le gaspillage mémoire lié à la fragmentation de 60-80% à moins de 4%, permettant de maximiser l'in-flight batching.

---

## 2. Dimensionnement VRAM : Combien gagne-t-on concrètement ?

La règle d'or du dimensionnement (Capacity Planning) repose sur la conversion du nombre de paramètres (Milliards / Billions) en octets. 

* **FP16 (16 bits) :** 1 paramètre = 2 octets. (Multiplicateur : x2)
* **FP8 / INT8 (8 bits) :** 1 paramètre = 1 octet. (Multiplicateur : x1)
* **INT4 (4 bits) :** 1 paramètre = 0.5 octet. (Multiplicateur : x0.5)

*Attention : Il faut toujours ajouter une marge de 15% à 20% pour le KV-cache et les activations.*

### Étude de cas sur GPU standards

Voici l'impact direct de la quantization sur le choix du matériel pour deux tailles de modèles populaires :

| Modèle cible | Précision (Format) | VRAM Poids seuls | VRAM Totale requise (avec contexte) | Matériel GPU minimum recommandé | Gain VRAM vs FP16 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **8B** (ex: Llama 3 8B) | **FP16** (Base) | ~16 Go | **~18 - 20 Go** | 1x RTX 3090/4090 (24 Go) ou 1x A10G | Réf. |
| **8B** | **FP8 / INT8** | ~8 Go | **~10 - 12 Go** | 1x RTX 4060 Ti (16 Go) ou T4 | **-50%** |
| **8B** | **INT4** (AWQ/GGUF) | ~4 Go | **~6 - 8 Go** | 1x RTX 3060 (12 Go) ou Laptop | **-75%** |
| **70B** (ex: Llama 3 70B)| **FP16** (Base) | ~140 Go | **~160 Go** | 2x H100 (80 Go) ou 2x A100 (80 Go)| Réf. |
| **70B** | **FP8 / INT8** | ~70 Go | **~78 Go** | **1x H100 (80 Go)** (*Économie massive*) | **-50%** |
| **70B** | **INT4** (AWQ/GGUF) | ~35 Go | **~42 Go** | 1x A6000 (48 Go) ou 2x RTX 3090/4090 | **-75%** |

### Le compromis : VRAM vs Vitesse vs Qualité

* **Gagner 50% de VRAM (Passage en FP8/INT8) :** La perte de qualité (Perplexité) est quasi indétectable (< 1%). Sur un H100, le FP8 **double la vitesse d'inférence** par rapport au FP16 grâce aux Tensor Cores spécialisés. C'est un gain absolu sur tous les tableaux.
* **Gagner 75% de VRAM (Passage en INT4) :** La dégradation qualitative devient mesurable (nuances de style perdues, légères hallucinations possibles sur des tâches de code/maths complexes). Le gain de vitesse est modéré car le calcul 4-bit n'est pas toujours accéléré nativement au niveau matériel.

---

## 3. Les Méthodes de Quantization Clés

Plusieurs approches Post-Training Quantization (PTQ) dominent l'écosystème :

* **SmoothQuant (INT8 W8A8) :** Lisse les valeurs extrêmes (outliers) des activations en transférant une partie de leur amplitude vers les poids via un rescaling. Permet une inférence 8-bit stable sans ré-entraînement, idéale sur architectures Ampere/Hopper.
* **AWQ (Activation-aware Weight Quantization - INT4) :** Identifie et protège le ~1% des poids les plus critiques (stockés en FP16), et quantifie les 99% restants en 4 bits. Offre le meilleur ratio qualité/compression pour des environnements contraints en VRAM.
* **GPTQ (INT3/INT4) :** Méthode one-shot utilisant l'approximation Hessienne pour compenser l'erreur de quantization bloc par bloc. Très populaire dans la sphère open-source.

---

## 4. Comparatif des Moteurs d'Inférence (Serving)

### A. TensorRT-LLM (NVIDIA)
Compilateur et runtime ultra-optimisé spécifique aux GPU NVIDIA.
* **Avantages :** Support FP8 natif sur H100, W8A8, AWQ, in-flight batching. Délivre le débit maximal et la latence minimale absolus (jusqu'à 4.6x plus de throughput sur H100 vs A100).
* **Inconvénients :** Verrouillage matériel (CUDA uniquement), nécessite une compilation (build engine) préalable.

### B. vLLM (Open-Source)
Serveur Python/C++ développé par UC Berkeley, standard de l'industrie.
* **Avantages :** Intégration PagedAttention native, support dynamique du FP8/INT8, chargement à la volée. API Python compatible OpenAI.
* **Inconvénients :** Latence pure sur une seule requête très légèrement supérieure à TRT-LLM, bien que le débit sous forte charge soit exceptionnel.

### C. llama.cpp / GGUF (CPU & Edge)
Implémentation minimaliste en C/C++ pour l'exécution locale.
* **Avantages :** Agnostique au matériel (CPU, Apple Silicon, petits GPU). Supporte des formats de quantization très granulaires (Q4_K_M, Q8_0).
* **Inconvénients :** Ne tire pas pleinement parti des Tensor Cores industriels. Déconseillé pour exploiter des serveurs data center.

---

## 5. Arbre de Décision : Quelle Stratégie Déployer ?

1.  **H100 End-to-End (Performance & Qualité Maximales)**
    * **Méthode :** FP8 (W8A8) + KV-Cache FP8.
    * **Outil :** TensorRT-LLM ou vLLM (argument: --quantization fp8).
    * **Bénéfice :** Permet de faire tenir un 70B sur un seul H100 avec une vitesse foudroyante et une qualité > 99% préservée.
2.  **Le Choix Universel (Robustesse sur A100 / GPU standards)**
    * **Méthode :** INT8 SmoothQuant (W8A8) + KV-Cache FP8.
    * **Outil :** vLLM (argument: --quantization int8).
    * **Bénéfice :** Idéal si le FP8 natif n'est pas supporté (ex: serveurs A100).
3.  **Haute Densité / Budget Réduit**
    * **Méthode :** INT4 AWQ (Poids uniquement) + KV-Cache FP8.
    * **Outil :** vLLM (argument: --quantization awq).
    * **Bénéfice :** Divise les coûts de VRAM par 4. Parfait pour les chatbots internes non-critiques où la fluidité prime sur le raisonnement complexe.

---

## 6. Bonnes Pratiques, Observabilité et Licences

### Monitoring et SLIs en Production
| Métrique Cible | Seuil d'Alerte (Exemple) | Action Corrective |
| :--- | :--- | :--- |
| Latence (Time To First Token) | p99 > 500ms | Scale-out ou bascule vers profil INT4. |
| Gaspillage KV-Cache | > 8% | Vérifier configuration PagedAttention. |
| Erreurs OOM | > 0.5% des requêtes | Réduire le contexte max ou KV en FP8. |

### Rappel sur la Conformité (Licensing)
La quantization est une transformation technique, elle **ne modifie en rien la licence source d'un modèle**.
* Un modèle "mergé" (ex: *Luminum-123B*) hérite systématiquement des restrictions de toutes ses composantes d'origine (ex: Mistral MRL + CC-BY-NC).
* **Règle d'or :** Si l'usage commercial ou la redistribution sont interdits sur le modèle de base, ils le restent sur le GGUF ou le moteur TensorRT. En cas de blocage, partagez uniquement les scripts de reproduction/merge.

---

## 7. Cheat-Sheet des Commandes de Déploiement

Pipeline vLLM FP8 (Recommandé sur Hopper/Ada) :

```bash
vllm serve mistralai/Mistral-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
```

Pipeline TensorRT-LLM (Build & Serve FP8) :

1. Export du Checkpoint HF :

```bash
python examples/llama/convert_checkpoint.py \
  --model_dir /path/to/model_hf \
  --output_dir /path/to/output_trtllm_ckpt \
  --dtype float16 --tp_size 1
```

3. Build de l'Engine FP8 :

```bash
trtllm-build \
  --checkpoint_dir /path/to/output_trtllm_ckpt \
  --output_dir /path/to/engine_fp8 \
  --use_fp8 --use_fp8_kv_cache \
  --max_batch_size 16 --max_input_len 8192
```

5. Serving :
trtllm-serve --engine_dir /path/to/engine_fp8 --port 8080


## 8. Références et Bibliographie

* **SmoothQuant (INT8 W8A8) :** Xiao et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*. ICML.
* **AWQ (INT4) :** Lin et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys.
* **GPTQ (INT3/INT4) :** Frantar et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR.
* **LLM.int8() :** Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS.
* **vLLM & PagedAttention :** Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP.
* **Format H100, FP8 & TensorRT-LLM :** Spécifications de l'architecture Hopper et documentation NVIDIA TensorRT-LLM.
