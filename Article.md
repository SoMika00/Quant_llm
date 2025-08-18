Guide pratique (2025) — Quantization LLM sur H100 et alternatives (TRT-LLM, vLLM, GGUF)
Les bases : tenseurs, poids, activations, KV-cache

Un tenseur est un tableau multi-dimensionnel de nombres (scalaires, vecteurs, matrices, etc.).

Dans un LLM :

Les poids (paramètres) sont stockés sur disque puis chargés en VRAM sous forme de tenseurs.

Les activations sont les résultats intermédiaires calculés pendant l’inférence.

Le KV-cache contient les Keys et Values de l’attention, conservés au fil de la génération pour accélérer l’auto-régression (évite de recalculer l’historique à chaque token).

Sur des serveurs modernes, la gestion du KV-cache est optimisée (Paged Attention, in-flight batching). Par exemple, PagedAttention stocke les K/V en blocs non contigus pour réduire le gaspillage mémoire.

Empreinte mémoire à l’inférence = poids du modèle + activations temporaires + KV-cache (taille qui croît avec la longueur de contexte).
En pratique, les poids dominent souvent (≈ 65% sur un 13B), le KV-cache ≈ 30% (selon longueur), les activations une part minime.
→ Taille minimale sur disque dictée surtout par les poids ; VRAM effective très sensible au KV-cache pour les contextes longs.

Overflow / Underflow (rappel)

Overflow : valeur > plage représentable → ∞.

Underflow : valeur trop petite → arrondie à 0.
Les formats à petite plage (FP8, INT4/INT8) nécessitent calibration (ex. SmoothQuant) pour lisser les outliers avant quantization.

Formats numériques : FP32, BF16, FP16, FP8, INT8, INT4

FP32 (32 bits) : précision élevée, grande dynamique. Référence entraînement, coûteux.

BF16 (16 bits) : 8 bits d’exposant (même dynamique que FP32), mantisse plus courte. Parfait en mixed-precision.

FP16 (16 bits) : plage réduite vs BF16, mais rapide (Tensor Cores), standard inférence.

FP8 (8 bits, E4M3 / E5M2) : introduit avec H100. Divise par 2 mémoire vs FP16, très rapide si bien calibré (choix E4M3/E5M2 par couche, per-tensor scaling).

INT8 : quantization entière. Deux options : poids-seuls ou W8A8 (poids + activations). SmoothQuant rend W8A8 stable.

INT4 : ultra-compressif (poids-seuls en pratique). Techniques AWQ / GPTQ pour limiter la perte. Gain mémoire fort, gain vitesse moindre (pas d’unités 4-bit natives).

Pourquoi préciser “poids + activations” ?
Beaucoup de quants ne touchent que les poids (taille disque/VRAM au chargement).
Mais si les activations restent en FP16, le calcul ne s’accélère pas autant.
→ W8A8 (ou FP8) tire parti de la quantization sur tout le chemin (GEMM, etc.).
→ Quantifier les activations est délicat (distributions variables) → calibration cruciale.

Résumé des formats (tableau)
Format	Bits	Exposant/Mantisse	Plage dynamique (approx.)	Utilisation typique & remarques
FP32	32	8 / 23	~1e-38 → 1e+38	Référence entraînement, calculs sensibles. Coûteux.
BF16	16	8 / 7	~1e-38 → 1e+38 (≈ FP32)	Mixed-precision entraînement/inférence. Même range que FP32, mantisse courte.
FP16	16	5 / 10	~1e-4 → 6.5e+4	Standard inférence GPU (Tensor Cores). Bon compromis.
FP8 E4M3	8	4 / 3	~1e-2 → ~4.5e+2	Inférence H100. Faible précision, calibration impérative.
FP8 E5M2	8	5 / 2	~1e-2 → ~5.7e+4 (+∞)	Plutôt pour grads/backward ou KV-cache plus large dynamique.
INT8 (W8A8)	8	entier (256 niv.)	via échelles (tensor/canal)	Poids+activations 8-bit. SmoothQuant pour stabilité. Tensor Cores int8.
INT4 (poids)	4	entier (16 niv.)	via échelles/groupes	Compression ×4 vs FP16. AWQ/GPTQ. Légère perte style/cohérence.
H100 et FP8 : ce qui change

La génération Hopper (H100) apporte des Tensor Cores FP8 + Transformer Engine (TE) qui gère automatiquement FP16↔FP8 selon un recipe optimal.

Bénéfices majeurs :

Débit ↑ (jusqu’à ×4–5 vs A100 FP16),

Latence 1er token ↓,

Mémoire ÷2 vs FP16,

Qualité ~FP16 si calibré (99%+ conservé dans des évaluations standard).

TRT-LLM exploite FP8 natif, in-flight batching, paged KV-cache → débits inédits sur H100.
En prod, certains gardent un profil FP16 de contrôle au début ; dans la pratique, FP8 bien calibré est quasi indiscernable.

Le KV-cache : FP16 vs FP8

Par défaut, KV en FP16 = fidélité max mais beaucoup de VRAM.
Passer le KV en FP8 ÷2 l’empreinte → plus de contexte et/ou plus de sessions.

Impact qualité très faible si calibration correcte.

Le dtype du KV est indépendant de celui des poids/activations :

ex. modèle FP16 + KV FP8 = quick win VRAM,

ou modèle FP8 + KV FP16 si on veut maximiser la précision des attentions (au prix de VRAM).

PagedAttention améliore l’usage KV quelle que soit la précision (pages petites, anti-fragmentation).
Combo gagnant multi-users : KV FP8 + PagedAttention.

Méthodes de quantization clés

SmoothQuant (INT8 W8A8, post-training).
Idée : lisser les outliers d’activation en transférant l’amplitude vers les poids via rescaling par couche → W8A8 stable sans fine-tuning.
Jusqu’à des modèles >500B avec perte négligeable. Gain mémoire ×2, accélération ~1.5×.

AWQ (INT4 poids-seuls, Activation-aware).
Repère ~1% de canaux critiques via les activations et les protège (8/16 bits) ; le reste en 4 bits.
Généralise bien (moins de sur-ajustement), qualité state-of-the-art en 4-bit, y compris instruction-tuned / multi-modal.

GPTQ (INT3/INT4 poids-seuls).
Quantization one-shot avec info de second ordre (approx. Hessienne) pour minimiser l’erreur.
Très rapide à appliquer, excellente qualité en 3–4 bits sur grands modèles.

LLM.int8 (bitsandbytes) (poids-seuls 8-bit).
Vectorisation par colonne + outliers traités en 16-bit → ~99.9% des ops en int8.
Allège la mémoire, n’accélère pas autant que W8A8/FP8 sur H100. Utile si pas de FP8/W8A8 natif.

Piles logicielles : TRT-LLM vs vLLM vs llama.cpp / GGUF
TensorRT-LLM (NVIDIA)

Compilateur/runtime qui génère un engine optimisé par GPU.

Forces : FP8 natif H100, INT8 (SmoothQuant), INT4 (AWQ), in-flight batching, paged KV, multi-GPU.

Perf absolues sur H100 au top.

Limites : spécifique NVIDIA, nécessite build ; support modèles modernes mais parser à jour requis ; engines non portables.

vLLM (Open-source)

Serveur haut débit avec PagedAttention (≈ <4% de mémoire KV gaspillée).

Supporte FP8, INT8 (W8A8), AWQ/GPTQ (chargement 4-bit), batching dynamique, streaming.

Très flexible (Python, OSS), perf excellentes (légèrement sous TRT-LLM à modèle égal), souvent meilleure efficacité en longue séquence / charge variable.

llama.cpp / GGUF (CPU & hétérogène)

Multiplateforme (CPU, Apple Silicon, petits GPU), déploiement simple (.gguf).

Multiples presets : Q8_0, Q4_K_M, etc.

Parfait pour prototypage, edge, machines sans CUDA.

Sur H100, préférer TRT-LLM/vLLM.

Compatibilité : engines TRT-LLM / modèles vLLM ≠ GGUF.
Toujours conserver le checkpoint HF original (les builds sont des dérivés).

“8-bit” sous vLLM : que choisir ?

Sur H100, choix par défaut : FP8 (W8A8).

vllm serve $MODEL_ID \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384


Alternative INT8 (W8A8) via SmoothQuant :

vllm serve $MODEL_ID \
  --quantization int8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384


Note : ne pas confondre avec load_in_8bit=True (bitsandbytes, poids-seuls).

Recommandations concrètes (cas 2×H100)

FP8 end-to-end (poids+acts FP8, KV FP8) = sweet spot H100 : qualité ~FP16, latence ↓, VRAM ↓.

INT8 SmoothQuant (W8A8) : stable, rapide, universel (légèrement moins rapide que FP8).

INT4 AWQ (+ KV FP8) : max densité (70B ≈ ~40 Go). Légère dégradation style/cohérence, à valider sur prompts sensibles.

Validation : calibrage + jeu de prompts réaliste, A/B blindé (FP16 vs quant), PPL, distinct-n ; ajuster repetition_penalty, temperature/top_p si besoin.

Pipelines type (déploiement)
Pipeline A — TRT-LLM FP8 (recommandé H100)

Exporter HF → TRT-LLM checkpoint :

python3 examples/llama/convert_checkpoint.py \
  --model_dir /models/YourModelHF \
  --output_dir /out/trtllm_ckpt \
  --dtype float16 --tp_size 2


Builder l’engine FP8 + KV FP8 :

trtllm-build \
  --checkpoint_dir /out/trtllm_ckpt \
  --output_dir /out/engine_fp8_tp2 \
  --tp_size 2 --max_batch_size 16 \
  --max_input_len 16384 --max_output_len 1024 \
  --use_fp8 --use_fp8_kv_cache


Servir :

trtllm-serve --engine_dir /out/engine_fp8_tp2 --port 8000

Pipeline B — TRT-LLM INT8 (SmoothQuant)

Export HF → TRT-LLM (idem).

Calibration SmoothQuant (dataset court).

Build INT8 (KV en FP8 recommandé) :

trtllm-build ... --quantize int8 --use_fp8_kv_cache ...


Serve via trtllm-serve.

Pipeline C — vLLM FP8 / INT8

FP8 :

vllm serve mistralai/Mistral-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192


INT8 (W8A8) :

vllm serve ORGANISATION/MODELE-HF \
  --quantization int8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384

Pipeline D — llama.cpp / GGUF (Q8_0 / Q4_K_*)

Conversion & quantization :

# Convertir HF -> GGUF FP16
python convert-hf-to-gguf.py NomDuModeleHF --outfile modele.gguf

# Quantifier en 4 bits
./quantize modele.gguf modele-q4_0.gguf q4_0

FAQ rapides

Pourquoi tous les modèles ne tournent pas en TensorRT-LLM ?
Besoin d’un parser/kernels spécifiques à l’archi. Les modèles standard (GPT/LLaMA…) sont supportés rapidement ; les variantes exotiques peuvent demander du délai. Engines NVIDIA-only.

Puis-je mixer modèle FP8 et KV FP16 ?
Oui. Poids/acts et KV ont des dtype indépendants. Choisir selon vos contraintes VRAM vs fidélité.

“8-bit” = FP8 ou INT8 ?
Les deux existent. Sur H100, 8-bit = souvent FP8 (meilleur ratio perf/qualité). Ailleurs : INT8 (SmoothQuant).

PagedAttention, c’est quoi ?
Gestion paginée du KV-cache pour éliminer la fragmentation et booster le throughput sous charge variable.

Et les quants GGUF (Q8_0, Q4_K_M, …) ?
Presets llama.cpp. Q8_0 ≈ très fidèle ; Q4_K_M = meilleur compromis 4-bit. Sur H100, préférer FP8/INT8 via TRT-LLM/vLLM.

Cas particulier : modèles “merge” & licences

Les merges héritent des restrictions les plus fortes des modèles sources (ex. NC, MRL).
Quantizer un modèle ne change pas la licence.
→ Ne pas redistribuer des poids quantifiés si la licence l’interdit.
Partager scripts (merge + quant) ou deltas/LoRA si autorisé.

Choisir sa quantization (arbre de décision)

Qualité ≈ FP16 + perfs max (H100) : FP8 (W8A8).
Stacks : TRT-LLM (perf max), vLLM FP8 (flex OSS). KV FP8 conseillé.

8-bit “classique” multi-plateformes : INT8 (SmoothQuant).
Stacks : vLLM --quantization int8, runtimes INT8 sur A100/CPU.

Compression agressive / VRAM limitée : INT4 (AWQ/GPTQ).
Sur H100, combiner INT4 poids + KV FP8.

Prototypage / Edge : GGUF (Q8_0, Q4_K, …) via llama.cpp.

Commandes types (référence rapide)

TRT-LLM (H100, FP8) :

# Export HF -> TRT-LLM ckpt
python examples/llama/convert_checkpoint.py \
  --model_dir /chemin/vers/modele_hf \
  --output_dir /chemin/vers/output_trtllm_ckpt \
  --dtype float16 --tp_size 2

# Build engine FP8 + KV FP8
trtllm-build \
  --checkpoint_dir /chemin/vers/output_trtllm_ckpt \
  --output_dir /chemin/vers/engine_fp8 \
  --use_fp8 --use_fp8_kv_cache \
  --max_batch_size 8 \
  --max_input_len 8192 --max_output_len 1024 \
  --tp_size 2

# Serve
trtllm-serve --engine_dir /chemin/vers/engine_fp8 --port 8080


vLLM FP8 (W8A8 + KV FP8) :

vllm serve ORG/MODELE-HF \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384


vLLM INT8 (W8A8 + KV FP8) :

vllm serve ORG/MODELE-HF \
  --quantization int8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384


GGUF (llama.cpp) :

python convert-hf-to-gguf.py NomDuModeleHF --outfile modele.gguf
./quantize modele.gguf modele-q4_0.gguf q4_0

Points de contrôle (qualité)

Jeu de validation réaliste (10–50 prompts, multi-tours, FR/EN).

Métriques : PPL, distinct-n, taux de refus/hallucinations.

A/B humain à l’aveugle (FP16 vs quant).

Tuning decoding : repetition_penalty ↑ (1.10→1.15), ajuster temperature/top_p.

Long contexte (≥16k) : tester KV FP8 ou FP16 selon VRAM/discernement.

TL;DR

H100 = FP8 natif → perf ×3–5, latence ↓, VRAM ÷2, qualité ~FP16 si calibré.

TRT-LLM = perf absolues (FP8/INT8, in-flight batching, paged KV).

vLLM = OSS très performant (PagedAttention, FP8/INT8, flexible).

Choix quant : FP8 si possible, sinon INT8 (SmoothQuant) ; INT4 (AWQ/GPTQ) pour densité.

GGUF (llama.cpp) = prototypage/edge, pas pour perf H100.

Licences : la quantization ne change pas la licence ; respecter NC/MRL ; publier scripts/deltas, pas les poids si restrictions.

Sources (liens regroupés)

docs.nvidia.com

developer.nvidia.com

nvidia.github.io

developers.redhat.com

blog.vllm.ai

runpod.io

arxiv.org

ar5iv.labs.arxiv.org

medium.com

qwen.readthedocs.io

huggingface.co
