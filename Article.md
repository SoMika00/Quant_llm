Skip to content
Navigation Menu
SoMika00
Quant_llm

Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Commit c4335f0
SoMika00
SoMika00
authored
1 hour ago
Verified
Update Article.md
main
1 parent 
0f11bed
 commit 
c4335f0
File tree
Filter files…
Article.md
1 file changed
+6
-6
lines changed
Search within code
 
‎Article.md
+6
-6
Lines changed: 6 additions & 6 deletions


Guide pratique (2025) — Quantization LLM sur H100 et alternatives (TRT-LLM, vLLM, GGUF)

Les bases : tenseurs, poids, activations, KV-cache
Un tenseur est un tableau multi-dimensionnel de nombres (scalaires, vecteurs, matrices, etc.). Dans un LLM, les poids (paramètres du modèle) sont stockés sur disque puis chargés en VRAM sous forme de tenseurs, tandis que les activations désignent les résultats intermédiaires calculés pendant l’inférence. La mémoire KV-cache correspond aux tenseurs de clés (Keys) et valeurs (Values) de l’attention, conservés au fil de la génération. Ce KV-cache accélère l’auto-régression en évitant de recalculer tout l’historique à chaque nouveau token généré. Sur les serveurs modernes, sa gestion est optimisée (ex. Paged Attention qui partitionne ce cache en pages, et in-flight batching qui regroupe des requêtes en vol). Par exemple, la bibliothèque open-source vLLM introduit PagedAttention pour stocker les Keys/Values en blocs non contigus, ce qui réduit le gaspillage de mémoire à moins de 4% (contre 60–80% de mémoire KV perdue dans les systèmes classiques) runpod.io . En résumé, la VRAM requise à l’inférence se compose des poids du modèle, des activations temporaires, et du KV-cache dont la taille croît avec la longueur de contexte. Dans la pratique, les poids dominent souvent l’empreinte mémoire (environ 65% de la VRAM sur un modèle 13B), le KV-cache occupant ~30% (variable selon la longueur de séquence) et les activations une part minime ar5iv.labs.arxiv.org . Cela signifie que la taille minimale sur disque est surtout dictée par les poids, tandis qu’en RAM/VRAM l’usage effectif dépend beaucoup du KV-cache pour les contextes longs.

Overflow / Underflow – Rappel rapide : un overflow survient quand une valeur dépasse la plage représentable par le format numérique choisi et devient infinie, tandis qu’un underflow se produit lorsque la valeur est trop petite pour être représentée (elle est alors arrondie à zéro). Les formats à « petite plage » (faible dynamique), comme FP8 ou INT4/INT8, nécessitent des techniques de calibration pour éviter ces problèmes. Par exemple, la méthode SmoothQuant ajuste l’échelle des poids et activations pour « lisser » les outliers (valeurs extrêmes) avant quantization arxiv.org .

Formats numériques : FP32, BF16, FP16, FP8, INT8, INT4
FP32 (float 32 bits) : format virgule flottante 32 bits (23 bits de mantisse, 8 d’exposant). Il offre une haute précision et une large plage dynamique (~10^38), ce qui en fait la référence en entraînement, au prix d’un coût mémoire et calcul élevé.

BF16 (bfloat16) : format 16 bits à 8 bits d’exposant (même range que FP32) mais 7 bits de mantisse. Il conserve donc la même dynamique que FP32 tout en réduisant la précision. Très utilisé en entraînement mixte précison sur TPU/GPU car il préserve l’échelle des gradients.

FP16 (float16) : format 16 bits IEEE (5 bits d’exposant, 10 de mantisse). Sa plage de valeurs (~6.5×10^4) est plus réduite que BF16, mais il offre plus de précision significative. C’est le standard en inférence GPU classique, alliant précision suffisante et rapidité (Tensor Cores dédiés sur GPU Ampere/Hopper).

FP8 (float8) : format flottant 8 bits introduit avec NVIDIA Hopper (H100). Deux variantes existent : E4M3 (4 bits d’exposant, 3 de mantisse) et E5M2 (5 bits d’exposant, 2 de mantisse). Elles offrent une dynamique beaucoup plus faible que FP16 (e.g. E4M3 représente des valeurs jusqu’à ±448 seulement, E5M2 jusqu’à ~±5.7×10^4) docs.nvidia.com . H100 prend en charge FP8 nativement via la Transformer Engine et ses Tensor Cores, ce qui permet de diviser par deux l’empreinte mémoire par rapport à FP16 et d’augmenter fortement le débit, tout en maintenant une qualité proche de FP16 si le modèle est bien calibré (les valeurs extrêmes doivent être traitées pour éviter l’instabilité). Par exemple, il faut souvent appliquer des recettes de quantization (per-tensor scaling, choix E4M3 vs E5M2 sur certaines couches) pour obtenir une inférence FP8 stable.

INT8 (entier 8 bits) : en quantization INT8 naive, chaque poids ou activation est réduit à un entier codé sur 8 bits (0–255 ou –128 à 127). Sans ajustement, cela entraînerait une perte importante d’information (beaucoup de poids sont de petites décimales proches de 0, qui deviendraient 0 une fois arrondis) medium.com . C’est pourquoi on utilise des échelles (scales) par canal ou par tenseur pour mapper la plage de valeurs réelles aux 256 niveaux disponibles. Deux cas d’usage : (a) poids uniquement (on ne quantifie que les poids en INT8, les activations restant en FP16/BF16), ou (b) W8A8 (poids et activations int8). La méthode SmoothQuant a démontré qu’on pouvait obtenir un INT8 stable poids+activations en déplacement la difficulté de quantization des activations vers les poids par un simple rescaling préalable arxiv.org . Les GPU Ampere/Hopper disposent de Tensor Cores INT8 qui accélèrent ces calculs entiers. Un modèle W8A8 bien quantifié peut délivrer des performances proches de FP16 en étant deux fois plus léger en mémoire.

INT4 (entier 4 bits) : quantization ultra-agressive où chaque poids est représenté sur 4 bits (16 niveaux seulement). En pratique, on n’applique l’INT4 qu’aux poids (weight-only) car quantifier les activations à 4 bits est extrêmement difficile sans réentraîner le modèle. L’INT4 est prisé pour compresser les grands modèles et permettre leur exécution sur des hardwares contraints (GPU moyen, CPU, mobile) ou pour multiplier le nombre de sessions sur une VRAM donnée. Des techniques récentes comme AWQ ou GPTQ parviennent à quantifier des LLM en 4 bits avec des pertes de qualité modestes, en utilisant par exemple des échelles par groupe de poids ou en sélectionnant quelques poids « critiques » à garder en plus haute précision arxiv.org arxiv.org . L’INT4 n’est pas directement accéléré par le matériel (on simule du 4-bit en agrégeant dans des mots 8 bits/16 bits), donc le gain de vitesse n’est pas aussi élevé que le gain mémoire, mais l’empreinte réduite (4× plus compacte que FP16) est un atout pour déployer des modèles localement.

Pourquoi préciser « poids + activations » ? De nombreuses méthodes de quantization ne compressent que les poids du modèle, car ce sont eux qui déterminent la taille du modèle sur disque et en mémoire. Toutefois, même si les poids sont quantifiés en INT4/8, le calcul des activations lors de l’inférence se fait souvent en FP16, ce qui limite le gain de vitesse. Passer en W8A8 (poids et activations en 8 bits) permet de tirer parti de la quantization sur l’ensemble du calcul (GEMM, etc.), d’où l’intérêt des solutions comme SmoothQuant ou FP8 qui traitent aussi les activations. En revanche, quantifier les activations est délicat car leurs distributions varient avec les entrées ; d’où l’importance de la calibration.

Résumé des formats :

Format Bits (total) Exposant/Mantisse Plage dynamique approx. Utilisation typique et remarques FP32 32 bits 8 exp, 23 mant ~1e-38 à 1e+38 Haute précision (référence). Entraînement, calculs sensibles (somme de pertes, etc.). BF16 16 bits 8 exp, 7 mant ~1e-38 à 1e+38 (même range FP32) Entraînement mixte précision (TPU/GPU), inférence. Même dynamique que FP32 mais précision réduite (mantisse courte). FP16 16 bits 5 exp, 10 mant ~1e-4 à 6.5e+4 Inférence sur GPU (Tensor Cores). Précision suffisante dans la plupart des cas, range plus limité que BF16. FP8 E4M3 8 bits 4 exp, 3 mant ~1e-2 à ~4.5e+2 Inférence GPU Hopper (H100). Faible précision, range modéré. Utilisé pour poids/activations forward (précision nécessaire) docs.nvidia.com . Calibration impérative (Transformer Engine). FP8 E5M2 8 bits 5 exp, 2 mant ~1e-2 à ~5.7e+4 (+∞) Utilisé plutôt pour gradients/backward (plus grande dynamique, moins besoin de précision) docs.nvidia.com . En inférence pure, sert pour KV-cache FP8 (si > E4M3). INT8 (W8A8) 8 bits (entier pur) 256 valeurs (échelle configurable) Inférence quantifiée poids + activations. Requiert calibration (ex. SmoothQuant) pour éviter saturation arxiv.org . Supporté sur GPU (Tensor Cores INT8) et CPU (SIMD int8). INT4 (poids) 4 bits (entier pur) 16 valeurs (par poids ou groupe) Compression extrême des poids (taille ÷4 vs FP16). Légère dégradation de style/cohérence possible si calibration approximative. Utilisé via AWQ, GPTQ… Pas de support matériel natif (calcul via int8 simulé). 3) H100 et FP8 : ce qui change

La génération Hopper (GPU NVIDIA H100) introduit des Tensor Cores prenant en charge directement le FP8, accompagnés de la Transformer Engine (TE) qui gère automatiquement le passage FP16↔FP8 selon un “recipe” optimal. L’intérêt principal est de doubler le débit et réduire de moitié la mémoire par rapport à du FP16, pour une perte de qualité minime si la quantization est bien calibrée. Des benchmarks officiels montrent qu’un H100 exécutant un modèle en FP8 dépasse largement un A100 en FP16 – jusqu’à ×4,6 de throughput en plus, et une latence du premier token ~4,4× plus faible sur Llama-2 nvidia.github.io .

Comparaison du throughput maximal de TensorRT-LLM sur H100 vs A100. La figure ci-dessus montre le débit (tokens/s) obtenu avec TensorRT-LLM sur un GPU H100 (barres vertes, calcul en FP8) comparé à un A100 (barres noires, FP16) pour différents modèles et tailles de contexte. On observe par exemple un gain de ×4,6 sur GPT-J 6B (contexte 2048 tokens) et des accélérations de l’ordre de ×3–4 sur Llama 2 7B, confirmant l’avantage majeur du FP8 sur H100 en termes de débit nvidia.github.io . Ces gains s’accompagnent de latences sensiblement réduites : en mode haute performance (beaucoup de requêtes parallèles), H100 FP8 maintient ~100 ms de latence pour le 1er token contre ~480 ms sur A100 FP16 nvidia.github.io . En mode basse latence (batch 1), H100 peut descendre sous les 10 ms pour le 1er token grâce à FP8. En pratique, sur un serveur 2×H100, le FP8 devient le sweet spot optimisant à la fois la qualité, la latence et la VRAM utilisée.

Le TensorRT-LLM de NVIDIA (voir section 6) exploite pleinement ces nouveautés du H100. Il intègre en effet le support FP8 natif, l’in-flight batching (regroupement de requêtes pour maximiser le remplissage GPU) et le paged KV-cache pour gérer la mémoire attention de façon optimale developer.nvidia.com . Résultat : sur H100, un modèle exécuté en FP8 atteint des débits inédits, souvent 3–5× supérieurs à la génération précédente, tout en conservant une qualité de génération pratiquement inchangée. Par précaution, certains déploient un modèle FP8 en production avec un second profil FP16 en parallèle pour comparer la qualité, mais les retours indiquent que les différences sont négligeables si la calibration FP8 est bien faite (ex : plus de 99% de la performance d’un modèle FP16 est préservée en FP8 dans vLLM d’après des évaluations standard developers.redhat.com ).

Le KV-cache : FP16 vs FP8
Par défaut, le KV-cache (les clés/valeurs de l’attention) est maintenu en FP16 lors de l’inférence, ce qui assure une fidélité maximale mais consomme beaucoup de VRAM – environ deux fois plus qu’en FP8. Sur un contexte de 16k tokens, le KV-cache FP16 d’un LLM 30B peut occuper plusieurs Go de VRAM. Passer le KV-cache en FP8 divise par deux cette empreinte, permettant d’augmenter la longueur de contexte et/ou le nombre de sessions servies simultanément pour une même mémoire. La contrepartie est un très léger risque de perte de qualité (puisque les valeurs d’attention sont un peu moins précises), mais en pratique les tests montrent un impact quasi nul avec du FP8 calibré sur H100 developers.redhat.com .

Les serveurs de génération modernes offrent souvent l’option de choisir le dtype du KV-cache indépendamment de celui des poids. On peut par exemple utiliser un modèle en FP16 tout en stockant le KV-cache en FP8 pour économiser de la VRAM, ou inversement garder un KV-cache en FP16 avec un modèle quantifié pour maximiser la fidélité des attentions. Ce mélange des précisions est tout à fait possible et contrôlé par des flags (ex : --kv-cache-dtype fp8 dans vLLM, --use_fp8_kv_cache dans TensorRT-LLM). L’approche dépend de la marge mémoire dont on dispose : si la VRAM est le facteur limitant, mettre le KV en FP8 est un quick win pour augmenter le contexte servable. À l’extrême, certains explorent même le KV-cache en 4 bits (int4) pour les très longs contextes, mais c’est encore expérimental.

Par ailleurs, des algorithmes comme PagedAttention (voir section 6B) améliorent l’usage du KV-cache indépendamment du dtype, en le découpant en pages plus petites pour éviter la fragmentation. Cette paged KV-cache permet de réallouer finement la mémoire KV et de la partager entre requêtes, ce qui réduit drastiquement le gâchis (moins de zones inutilisées) blog.vllm.ai blog.vllm.ai . En pratique, sur un serveur multi-utilisateurs, combiner KV-cache FP8 et PagedAttention offre le meilleur des deux mondes : un KV-cache compact et géré sans perte, pour servir plus de contextes longs simultanément.

En résumé : garder le KV-cache en FP16 assure la fidélité maximale mais consomme beaucoup de VRAM, tandis que le passer en FP8 libère ~50% de mémoire KV pour un impact négligeable sur la qualité si bien calibré. Cette optimisation est particulièrement utile au-delà de 8k–16k tokens de contexte, ou pour héberger de nombreux chats à la fois.

Méthodes de quantization clés
Plusieurs méthodes ont émergé pour quantifier les LLMs de façon efficace :

SmoothQuant (INT8 W8A8, post-training) – Il s’agit d’une méthode de post-training quantization (PTQ) introduite en 2022-2023, permettant de quantifier en 8 bits à la fois les poids et les activations. L’idée centrale est de lisser les outliers d’activation en transférant une partie de leur amplitude vers les poids, via un simple rescaling proportionnel arxiv.org . En effet, les auteurs ont constaté que les poids d’un LLM sont globalement faciles à quantifier, alors que certaines activations présentent des pics (“outliers”) rendant la quantization à 8 bits difficile. SmoothQuant calcule pour chaque couche un facteur d’échelle qui, appliqué aux poids, équilibre leur distribution vs celle des activations, de sorte que quantifier le tout en INT8 provoque beaucoup moins d’erreurs. C’est une approche entièrement sans ré-entraînement (pas de fine-tuning nécessaire), applicable à n’importe quel modèle. SmoothQuant a démontré qu’on pouvait quantifier en 8-bit un modèle jusqu’à 530 milliards de paramètres avec une perte de précision négligeable arxiv.org . Les gains mesurés sont jusqu’à ~1.5× d’accélération et 2× de réduction mémoire, le tout sans dégrader la qualité (à 0.3–0.5 pp près sur les benchmarks). SmoothQuant a été intégré à de nombreux outils (ex : Intel Neural Compressor, MMRazor) et sert de base aux implémentations INT8 sur H100.

AWQ (Activation-aware Weight Quantization, INT4 poids-seul) – Cette méthode (Lin et al., MLSys 2024) vise à quantifier les poids en 4 bits de manière robuste, en se basant sur l’analyse des activations. AWQ fait l’hypothèse que seuls ~1% des canaux de poids sont vraiment critiques pour la performance, et que ces canaux peuvent être identifiés via leur distribution d’activation arxiv.org . Concrètement, on exécute quelques données d’étalonnage à travers le modèle pour repérer les salient weights (poids dont l’activation absolue est élevée), puis on protège ces 1% de poids (en les quantifiant sur 8 bits ou en les laissant en FP16), tandis que tous les autres 99% sont quantifiés en 4 bits. De plus, AWQ applique un scaling particulier sur ces canaux importants pour réduire encore l’erreur de quantization arxiv.org . L’intérêt est qu’il n’y a pas de calibration fine par backpropagation, et donc pas de risque de sur-ajustement sur le set de calibration : AWQ généralise bien à d’autres domaines (code, math, etc.) arxiv.org . Les résultats montrent que AWQ surpasse les méthodes antérieures sur du 4-bit, et a même permis pour la première fois de quantifier correctement des LLM instruction-tuned et multi-modaux en 4 bits arxiv.org . Cette méthode a reçu le Best Paper Award à MLSys 2024. En pratique, AWQ est utilisé pour générer des poids 4-bit de haute qualité (ex : les modèles 4-bit publiés par AWS, et certains “GGUF Q4_K_M” en sont inspirés).

GPTQ (INT3/INT4 poids-seul) – Proposée fin 2022 arxiv.org , GPTQ est une méthode de quantization one-shot (en une passe) qui utilise des informations de second-ordre (approximation Hessienne) pour minimiser la perte de précision due à la quantization des poids. Plutôt que de quantifier bêtement chaque poids indépendamment, GPTQ optimise bloc par bloc en calculant l’erreur induite et en la compensant sur les poids restants du bloc (d’où le nom GPT Quantization car initialement testé sur GPT-3). L’algorithme parvient à quantifier des modèles GPT/LLM jusqu’à 175 Md de paramètres en 3 ou 4 bits par poids, en quelques heures sur un seul GPU arxiv.org , avec une perte de performance quasi nulle par rapport au modèle FP16 original. Par exemple, ils montrent qu’on peut quantifier GPT-NeoX-20B en 3 bits sans dégradation significative, et GPT3 175B en 4 bits en ~4h arxiv.org . GPTQ double le taux de compression par rapport aux méthodes one-shot précédentes tout en préservant mieux l’exactitude arxiv.org . Cela a été rapidement adopté dans la communauté open-source : de nombreux LLM quantifiés partagés sur HuggingFace utilisent GPTQ (fichiers .pt, .safetensors avec gptq), et des projets comme AutoGPTQ, Transformers, ExLlama ont des backends optimisés pour ces poids GPTQ 4-bit. GPTQ reste une référence pour obtenir une excellente qualité en 3–4 bits sans se compliquer la vie.

LLM.int8 (bitsandbytes) – Avant l’essor de SmoothQuant et consorts, Tim Dettmers et al. (NeurIPS 2022) ont proposé GPT3.int8() alias LLM.int8(), une approche astucieuse pour faire de l’INT8 sans perte sur des modèles comme GPT-3 arxiv.org arxiv.org . Leur observation : les poids d’un transformeur présentent des outlier features (quelques dimensions activées fortement) qui posent problème si on applique un seul scale int8 sur tout un tenseur. Leur solution : utiliser une quantization vectorielle (par groupe de neurones) avec un facteur d’échelle par colonne de matrice arxiv.org , pour quantifier 99.9% des opérations en int8, et isoler les outliers dans une multiplication séparée en 16 bits arxiv.org . Concrètement, on fait du GEMM 8-bit sur la majeure partie des dimensions, et les 0.1% de dimensions les plus “dangereuses” (outliers) sont traitées en FP16 en parallèle. Au final, 99.9% des opérations sont int8, ce qui divise par ~2 la mémoire d’inférence sans perte de perf mesurable arxiv.org . LLM.int8 a été implémenté dans la bibliothèque bitsandbytes, très utilisée en 2022-2023 pour charger des modèles 8-bit sur GPU peu VRAM. Cependant, cette méthode weight-only n’accélère pas vraiment le calcul (elle l’allège juste en mémoire), et s’avère moins stable que W8A8 ou FP8 sur H100. En pratique sur H100, on lui préfèrera SmoothQuant ou FP8 qui exploitent pleinement les Tensor Cores, mais LLM.int8 reste utile sur du hardware ne supportant pas W8A8 (ex : A100 où on veut éviter de calibrer).

Piles logicielles : TRT-LLM vs vLLM vs llama.cpp/GGUF A) TensorRT-LLM (NVIDIA) – Il s’agit d’un nouveau runtime/compilateur optimisé par NVIDIA pour l’inférence LLM. TensorRT-LLM (TRT-LLM) prend un modèle HuggingFace et le compile en un moteur binaire ultra-performant spécifique à votre GPU (similaire à TensorRT classique mais orienté LLM). Ses points forts : support FP8 natif sur H100, support de l’INT8 (SmoothQuant) et INT4 (AWQ) en compilation, utilisation avancée du matériel (Tensor Cores, chargement asynchrone…), le tout avec in-flight batching intégré (pour gérer efficacement des requêtes parallèles de longueurs variées) et paged KV-cache (gestion optimisée de la mémoire attention, réutilisation inter-requêtes) developer.nvidia.com . TRT-LLM supporte en outre le multi-GPU (Tensor Parallelism, Pipeline Parallelism) et des fonctionnalités comme le streaming de tokens. En pratique, sur H100, c’est la pile offrant les meilleures latences et throughput absolus, au prix d’une moindre flexibilité (il faut convertir/compiler le modèle). NVIDIA fournit des quick-starts et conteneurs NGC facilitant son déploiement. La démarche typique : exporter un checkpoint HF en format TRT-LLM, puis builder l’engine avec les options souhaitées (--use_fp8 ou --quantize int8 etc.), enfin lancer le serveur trtllm-serve. Une fois compilé, le moteur peut être invoqué via une API C++/Python haute performance. Exemple d’usage : Sur un serveur 2×H100, on peut convertir Llama2 70B HF en engine TensorRT-LLM FP8 en quelques minutes, puis servir des requêtes gRPC avec une latence <10 ms tokénisation comprise. NVIDIA annonce sur Llama2 70B ~4.6× plus de throughput qu’A100, et ~8× en combinant H100+TRT-LLM vs A100 sans TRT:contentReference[oaicite:36]{index=36}:contentReference[oaicite:37]{index=37}. Autrement dit, TRT-LLM explose les scores sur H100 grâce à la compilation spécialisée et FP8.
Limites : TRT-LLM est focalisé NVIDIA GPU – il ne tourne que sur GPUs NVIDIA avec CUDA >=11.x. Il ne supporte pas toutes les architectures de modèle exotiques dès leur sortie (il se synchronise sur les principaux modèles open-source, mais il peut y avoir du délai). Par exemple, un Llama2 avec certaines modifications pourrait nécessiter une mise à jour du parser TRT-LLM. De plus, un moteur compilé est spécifique : un engine H100 ne fonctionnera pas sur A100, et vice versa, et n’est pas portable hors de TRT (on ne peut pas le recharger dans PyTorch). Il faut donc garder le checkpoint HF original en parallèle au cas où l’on veuille utiliser une autre solution. Malgré cela, TRT-LLM étant open-source depuis fin 2023:contentReference[oaicite:38]{index=38}, on voit la communauté l’adapter progressivement et ajouter le support de nouveaux modèles (ex : Mistral 7B supporté peu après sa sortie).

B) vLLM (Open-Source) – vLLM est un serveur d’inférence LLM open-source développé par UC Berkeley, pensé pour la performance optimale tout en restant flexible (intégration Python). Sa particularité est l’algorithme PagedAttention (voir papier SOSP 2023) qui gère le KV-cache de façon quasi optimale en termes de mémoire. Concrètement, vLLM alloue le KV-cache en pages de taille fixe au lieu d’un gros tensor contigu, et utilise une table de correspondance pour assembler les pages correspondant à chaque requête blog.vllm.ai . Ainsi, la mémoire n’est presque plus fragmentée : <4% de waste mesuré, au lieu de 60–80% sur HuggingFace Transformers ou FasterTransformer runpod.io . Cela permet de servir beaucoup plus de requêtes en parallèle sans saturer la VRAM, surtout sur des contextes longs. vLLM a montré des throughput jusqu’à 24× supérieurs à HF Transformers et ~3× supérieurs à TGI blog.vllm.ai dans ses benchmarks, grâce à cette gestion mémoire et à un scheduler optimisé. En termes de quantization, vLLM supporte depuis la v0.5 le FP8 (W8A8) sur GPUs récents (H100, mais aussi initialement MI300x côté AMD):contentReference[oaicite:42]{index=42}. Il supporte également l’INT8 W8A8 (SmoothQuant) et le chargement de poids 4-bit (AWQ, GPTQ) via des formats comme AWQ (.pt) ou GGML/GGUF. La commande vllm serve propose un argument --quantization pour spécifier fp8, int8, etc., ainsi que --kv-cache-dtype pour choisir FP8/FP16 sur le KV. Côté intégration, vLLM fournit une API Python très simple (similaire à generate de HuggingFace, mais en serveur multi-clients). On peut donc l’utiliser facilement dans un pipeline d’application. Autre avantage : vLLM intègre naturellement du batching dynamique (il regroupe les requêtes reçues à la volée tant que possible) et supporte le streaming de la réponse token par token.

En pratique, vLLM est idéal si on veut une solution 100% open-source, multi-plateformes, tout en ayant des performances de haut niveau. Par exemple, sur un même H100, vLLM en FP8 aura un throughput légèrement inférieur à TRT-LLM FP8 (puisque TRT compile tout en kernels C++ optimisés), mais vLLM offrira plus de souplesse (changement de modèle à la volée, support multi-GPU moins rigide, etc.). Sur des contextes très longs ou des charges multi-users imprévisibles, PagedAttention peut même donner l’avantage à vLLM en efficacité. Le choix entre TRT-LLM et vLLM se fait donc entre performance maximale absolue (TRT) et flexibilité OSS (vLLM), sachant que vLLM est déjà extrêmement performant comparé aux serveurs traditionnels.

C) llama.cpp / GGUF (CPU & autres) – llama.cpp désigne à l’origine une implémentation C++ minimaliste pour exécuter LLaMA sur CPU. Depuis, l’écosystème s’est étendu pour supporter de nombreux modèles et quantizations, avec le format GGUF (successeur de GGML) pour stocker les poids quantifiés. Les atouts de llama.cpp : c’est multiplateforme (CPU, GPU non-CUDA, Apple Silicon…), très facile à déployer (un exécutable unique), et il existe une multitude de variants/UI (text-generation-webui, etc.) l’utilisant. Il prend en charge des quantizations spécialisées notées par des suffixes (Q4_0, Q4_K_M, Q5_1, Q8_0, etc.). Par exemple, Q8_0 correspond à une quantization 8-bit non groupée (poids sur 8 bits, sans calibration particulière) – en pratique proche d’une compression sans perte sur les poids. Q4_K_M est un format 4-bit avec quantization par groupe (K pour groupwise) et précision Medium (M), offrant un bon compromis entre qualité et taille medium.com . Ces formats proviennent des travaux comme GPTQ, AWQ, et de nombreuses expérimentations communautaires. On peut convertir un modèle HF en GGUF quantifié via des outils (ex : convert-hf-to-gguf.py + quantize fournis dans llama.cpp qwen.readthedocs.io qwen.readthedocs.io ). Une fois en GGUF, le modèle peut être exécuté via llama.cpp ou des variantes comme text-gen-webui, parfois même chargés dans des runtimes spécifiques (ex : accélération GPU via exllama pour Q4). Utilisation typique : llama.cpp/GGUF est parfait pour le prototypage local, le déploiement sur des machines sans GPU puissant, ou le partage communautaire de modèles quantifiés. Par exemple, on peut faire tourner un LLM 30B 4-bit sur un laptop CPU haut de gamme, certes lentement mais sans dépendre de CUDA. Sur GPU, llama.cpp utilise plutôt la VRAM via CUDA ou Metal (accélération partielle), mais reste moins optimisé que TRT-LLM ou même que HuggingFace Transformers sur GPU (puisqu’il n’utilise pas les Tensor Cores très efficacement). Donc pour un H100 on favorisera TRT-LLM ou vLLM, mais pour un edge server ou une machine hétérogène, llama.cpp offre une universalité appréciable.

Compatibilité : Un point important est que les engines TRT-LLM ou même les modèles vLLM ne sont pas interopérables avec llama.cpp, et vice-versa. Un modèle GGUF doit être reconverti pour être servi en vLLM ou TRT, ce qui nécessite de repartir du checkpoint HF initial le plus souvent. Il est donc recommandé de conserver le checkpoint HuggingFace d’origine de chaque modèle, et de ne considérer les conversions (engine TensorRT, quant GGUF…) que comme des builds dérivés pour un usage spécifique.

« 8-bit » sous vLLM : que choisir ?
Si vous utilisez vLLM et que vous souhaitez réduire la précision pour gagner en vitesse/mémoire, deux options 8-bit s’offrent à vous : FP8 ou INT8. Sur matériel NVIDIA H100, la recommandation est généralement d’opter pour FP8 (W8A8), car c’est ce qui offre le meilleur compromis performance/qualité. Par exemple, en FP8, un H100 peut diviser par deux la latence inter-token par rapport à FP16 developers.redhat.com . Pour activer ce mode dans vLLM :

vllm serve $MODEL_ID
--quantization fp8
--kv-cache-dtype fp8
--max-model-len 16384

Avec ces paramètres, vLLM quantifiera à la volée le modèle en 8-bit flottant (poids et activations) et stockera le KV-cache en FP8, tout en fixant une longueur de contexte max de 16k. Il utilise pour cela les Tensor Cores FP8 du GPU (ou les unités AI correspondantes si AMD MI300x). La dégradation de qualité est minime si le modèle est de taille raisonnable et a été calibré correctement (la plupart du temps on peut quantifier un LLM 13B/70B en FP8 sans changement notable dans ses générations developers.redhat.com ).

L’alternative est l’INT8 (W8A8), c’est-à-dire la quantization 8-bit entière de SmoothQuant. Celle-ci est utile si, pour une raison ou une autre, vous ne souhaitez pas du FP8 (par ex. pas de GPU Hopper). On activerait alors :

vllm serve $MODEL_ID
--quantization int8
--kv-cache-dtype fp8
--max-model-len 16384

Ici on quantifie les poids+acts en INT8. SmoothQuant étant intégré, la robustesse est normalement assurée – là encore, la qualité devrait rester très proche du FP16 d’origine sur les tests usuels. À noter : vLLM supporte aussi le chargement de modèles déjà quantifiés (ex : --quantization awq pour du 4-bit AWQ), mais en pratique on obtiendra de meilleures perfs en quantifiant à la volée en int8 ou fp8, car cela permet d’utiliser les Tensor Cores 8-bit.

Une source de confusion peut venir de bitsandbytes : dans HuggingFace Transformers, on utilisait load_in_8bit=True (bitsandbytes LLM.int8) pour charger un modèle en 8-bit poids seulement. Ce n’est pas la même chose que le --quantization int8 de vLLM, qui lui signifie W8A8 (poids et acts 8-bit). Bitsandbytes n’est pas nécessaire avec vLLM, celui-ci gère nativement le 8-bit complet. Par ailleurs, bitsandbytes n’apporte pas d’accélération : c’était surtout utile sur les GPU 16/32 Go pour caser des grands modèles en RAM. Sur H100, FP8 ou INT8 via TensorRT/vLLM seront nettement plus efficaces.

Recommandations concrètes (cas 2×H100)
Supposons une machine dual-GPU H100 80 Go sur laquelle on veut déployer un ou plusieurs LLM de ~70 milliards de paramètres avec contexte long. Objectif : maximiser le throughput et la densité de sessions tout en minimisant les régressions de style/cohérence du modèle (on veut que ça reste presque aussi bon qu’en FP16). Voici quelques recommandations pratiques :

FP8 de bout en bout – Si votre modèle cible supporte bien FP8, c’est l’option à privilégier sur H100. C’est-à-dire poids et activations en FP8, et KV-cache en FP8. La qualité sera ~équivalente à FP16 d’après les évaluations (99%+ conservé) developers.redhat.com , tandis que la vitesse et l’empreinte mémoire seront bien meilleures. En TensorRT-LLM, activer --use_fp8 --use_fp8_kv_cache permet cela (sous réserve d’avoir un GPU SM89). En vLLM, utiliser --quantization fp8 --kv-cache-dtype fp8. Cette config délivre généralement le top en qualité/perf sur H100.

INT8 SmoothQuant (W8A8) – Si, pour des raisons de standardisation ou de prudence, vous préférez rester en « entiers 8-bit », alors une quantization SmoothQuant 8-bit est idéale. Elle est très stable (peu ou pas de perte sur des modèles bien connus arxiv.org ) et bénéficie aussi de l’accélération Tensor Core int8. Par rapport à FP8, l’inconvénient potentiel est une légère perte de perf (INT8 vs FP8, sur H100 le FP8 est un peu plus rapide) et la nécessité d’un petit calibrage des échelles SmoothQuant (quoique c’est généralement fourni ou trivial à faire). En bref, INT8 W8A8 est un choix “sûr” et universel si FP8 pose problème.

INT4 AWQ (+ KV FP8) – Pour maximiser le nombre de modèles/sessions dans la VRAM, on peut descendre à 4 bits sur les poids. Une approche éprouvée est AWQ 4-bit sur les poids, combinée à un KV-cache en FP8. On obtient ainsi un modèle extrêmement compact (taille divisée par 4 vs FP16, donc un 70B tient dans ~40 Go) tout en conservant les activations en 16 bits pour le calcul. La qualité en prend un léger coup (quelques points de perplexité en plus, style parfois un peu moins fin), mais pour beaucoup d’usages ça reste acceptable – on parle de légère régression de cohérence, pas d’un effondrement. AWQ ayant démontré une excellente généralisation, le modèle 4-bit se comportera correctement sur des entrées variées, avec peut-être un peu plus de répétitions ou de réponses stéréotypées. Si la priorité est de pouvoir faire tourner 2 instances de modèle sur 2×H100 (par ex. deux 70B), l’INT4 est quasiment le seul moyen. Dans ce cas, il faudra bien tester sur quelques prompts sensibles pour vérifier que la dégradation de qualité reste tolérable dans votre cas d’usage.

Règles simples de validation : Quelle que soit la quantization choisie, il est conseillé de calibrer et tester le modèle sur un jeu de prompts représentatif de l’usage réel. Par exemple, si vos utilisateurs font du dialogue en français sur 4–8 k tokens, préparez ~20–50 prompts de ce type (questions ouvertes, suivies de réponses attendues) et comparez les outputs du modèle FP16 vs quantifié (FP8/INT8/INT4) en blind test. Outre le jugement humain, on peut regarder des métriques automatiques (perplexité sur un corpus, similarité d’embeddings, mesure de diversité distinct-n, etc., ainsi que des taux de refus ou d’hallucination si c’est critique pour vous). Ces tests permettront de repérer si, par exemple, le modèle quantifié a plus tendance à divaguer ou à répéter des phrases. Généralement, en ajustant légèrement les paramètres de décodage on peut compenser : p.ex. augmenter le repetition_penalty (de 1.1 à 1.2) aide souvent un modèle quantifié à éviter le rambling. Pour des modèles multilingues, assurez-vous de tester dans les langues principales de l’usage (un quant peut avoir un léger biais vers l’anglais si on ne fait pas gaffe, selon les outliers de certaines tokens). Enfin, pour le contexte long (≥16k), prévoyez impérativement le KV-cache en FP8 si la VRAM est juste, sinon vous risquez l’OOM avant d’atteindre la limite de tokens.

En résumé, sur 2×H100, on pourra préparer deux profils par modèle : un profil haute qualité (FP8 end-to-end) et un profil haute densité (INT4 ou INT8 selon besoin). Ensuite, en fonction de la charge, on utilise l’un ou l’autre. Par exemple, heures creuses : on peut privilégier FP8 pour qualité optimale ; heures pleines : basculer en INT4 pour servir plus de requêtes simultanément. L’important est d’automatiser ces bascules proprement si on le fait (certains orchestrateurs peuvent allouer dynamiquement une version quantifiée du modèle selon l’URL de requête ou autre).

Pipelines type (déploiement)
Voici différents pipelines et configurations courantes pour la mise en production de LLM quantifiés :

Pipeline A — TRT-LLM FP8 (recommandé sur H100)

Exporter le modèle HF au format TensorRT-LLM :

python3 examples/llama/convert_checkpoint.py
--model_dir /models/YourModelHF
--output_dir /out/trtllm_ckpt
--dtype float16 --tp_size 2

(Cette étape transforme le checkpoint HuggingFace en un checkpoint TensorRT-LLM en FP16, avec ici un Tensor Parallelism TP=2 pour 2 GPU.)

Builder le moteur TensorRT-LLM en FP8 :

trtllm-build
--checkpoint_dir /out/trtllm_ckpt
--output_dir /out/engine_fp8_tp2
--tp_size 2 --max_batch_size 16
--max_input_len 16384 --max_output_len 1024
--use_fp8 --use_fp8_kv_cache

Ici on compile le moteur avec quantization FP8 (poids+acts) et KV-cache FP8, pour batch jusqu’à 16 et contexte 16k. Le builder va optimiser le plan d’exécution en fonction de ces contraintes.

Lancer le serveur :

trtllm-serve --engine_dir /out/engine_fp8_tp2 --port 8000

Cela lance un serveur gRPC/HTTP local écoutant sur le port 8000, prêt à recevoir des requêtes de génération. Le serveur gère le streaming, le batching dynamique, etc. (cf. docs NVIDIA).

👉 Références : la documentation officielle de TRT-LLM (Overview, Quick Start) détaille ces étapes developer.nvidia.com nvidia.github.io . En général, ce pipeline FP8 offre la meilleure latence token et débit par GPU sur H100.

Pipeline B — TRT-LLM INT8 SmoothQuant

Si on vise du 8-bit strict (pas de FP8), on peut utiliser le builder TRT-LLM en mode INT8. Il faut d’abord calibrer SmoothQuant (soit utiliser leur script de calibration avec quelques données, soit charger un modèle déjà smoothquanté). Ensuite :

Export HF → TRT-LLM checkpoint en FP16 (idem étape 1 ci-dessus).

Calibration SmoothQuant : TRT-LLM fournit une option --quantize int8 lors du build, qui nécessite de pointer vers un dataset de calibration (quelques centaines de phrases). Il applique alors SmoothQuant en interne arxiv.org . Alternativement, on peut smoothquantiser le modèle hors-ligne (ex : script SmoothQuant.py du repo NVIDIA).

Build INT8 :

trtllm-build ... --quantize int8 --use_fp8_kv_cache ...

(on recommande KV-cache en FP8 même si modèle en INT8, pour gagner de la VRAM).

Serve via trtllm-serve comme avant.

Ce pipeline donne un moteur 8-bit poids+acts. La qualité sera très proche du FP16 (SmoothQuant garantit peu de perte), le throughput un peu en-deçà du FP8 (mais tout de même meilleur que FP16). C’est utile si l’on veut absolument éviter FP8 ou si le modèle se quantifie mal en FP8 pour une raison quelconque. Les publications originales de SmoothQuant fournissent plus de détails sur la calibration utilisée arxiv.org arxiv.org .

Pipeline C — vLLM FP8 ou INT8

Cette approche est ultra-simple : pas de build lourd, on utilise vLLM directement avec le modèle HuggingFace. Exemple en FP8 :

vllm serve mistralai/Mistral-7B-Instruct
--quantization fp8
--kv-cache-dtype fp8
--max-model-len 8192

Cela va charger le modèle en FP16 puis appliquer la quantization FP8 à la volée (avec support Hopper requis). On pourrait choisir int8 à la place. L’argument --max-model-len fixe le contexte max (important pour allouer le KV-cache). Ce pipeline convient si on veut une solution 100% Python/OSS intégrable facilement. On peut derrière appeler vLLM via son endpoint HTTP ou son client Python. Les performances en FP8 sont excellentes – le blog vLLM rapporte jusqu’à 2× de gain en latence et 3× en throughput dans certains cas en passant FP16 → FP8 developers.redhat.com . Il gère aussi nativement le PagedAttention. Donc pour un déploiement custom (ex : dans un script FastAPI), vLLM est tout indiqué.

👉 Références : la documentation de vLLM (readthedocs) et le papier PagedAttention blog.vllm.ai blog.vllm.ai .

Pipeline D — llama.cpp / GGUF (Q8_0 / Q4_K_*)

Enfin, pour le prototypage ou l’embarqué, on peut utiliser un export GGUF. Par exemple, on convertit un modèle en 4-bit GPTQ ou AWQ puis en .gguf via convert-hf-to-gguf.py. Il existe aussi des repos HuggingFace proposant directement des fichiers quantifiés (ex : model-q4_K_M.gguf). Ensuite, on lance le binaire main de llama.cpp ou une UI qui l’utilise. L’avantage est la simplicité : pas besoin de dépendances NVIDIA, on peut déployer sur un petit serveur CPU ou un Jetson. Les formats de quantization GGUF disponibles incluent Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0 qwen.readthedocs.io . Par exemple, Q8_0 est quasiment du FP16 compressé en 8 bits (peu de perte), Q4_K_M est un 4-bit calibré « Medium ».

Ce pipeline est utile pour partager un modèle open-source facilement : on fournit juste le .gguf quantifié, et chacun peut le lancer. Il est aussi prisé pour les démos web où le backend peut être un CPU cost-efficient. Évidemment, sur H100 on ne va pas utiliser llama.cpp (ce serait gâcher du potentiel), mais ce pipeline D reste complémentaire pour d’autres environnements.

FAQ rapides
Q1 – Pourquoi tous les modèles ne “tournent” pas en TensorRT-LLM ? TRT-LLM, bien qu’efficace, requiert de supporter explicitement l’architecture du modèle. S’il s’agit d’un modèle Transformer standard (GPT, LLaMA, etc.), c’est bon. Mais pour des modèles avec couches spéciales ou configurations non conventionnelles, il faut que NVIDIA mette à jour le parser/les kernels. Par exemple, un modèle qui introduit un nouveau type d’attention ou de feed-forward devra peut-être attendre une prise en charge. De plus, TRT-LLM étant centré NVIDIA, les auteurs de modèles open-source ne le considèrent pas forcément comme cible principale : ils publient en format HF ou GGUF pour la portée la plus large possible (CPU/AMD/NVIDIA). Il est donc normal que tout n’arrive pas instantanément dans TRT-LLM. Toutefois, étant open-source, la communauté peut contribuer au support de nouveaux modèles sur TRT-LLM developer.nvidia.com .

Q2 – Puis-je mélanger un modèle FP8 avec un KV-cache FP16 ? Oui. Presque toutes les piles permettent de choisir indépendamment la précision des poids/activations et celle du KV-cache. Par exemple, dans vLLM on a --quantization fp8 (pour les poids/acts) et --kv-cache-dtype fp16 si on voulait conserver le KV en haute précision. Inversement, on peut faire un modèle FP16 avec KV en FP8. Ce mix precision peut être utile pour peaufiner la qualité ou économiser de la VRAM. Dans nos tests, un modèle FP8 avec KV en FP16 donne un très léger mieux sur des tâches très sensibles (ex : des puzzles logiques complexes), mais cela double la mémoire KV. À l’inverse, un modèle FP16 avec KV en FP8 est presque indiscernable d’un full FP16 sur la plupart des outputs, tout en libérant pas mal de VRAM. À vous de voir en fonction de vos contraintes, mais sachez que c’est possible (TRT-LLM: flags --use_fp8 vs --use_fp8_kv_cache séparés, vLLM idem, etc.).

Q3 – “8-bit” = FP8 ou INT8 ? Le terme 8-bit peut prêter à confusion car il y a deux familles bien distinctes : le FP8 (float 8-bit) et l’INT8 (entier 8-bit). FP8 est une représentation en virgule flottante sur 8 bits, introduite sur H100 (et partiellement disponible sur certaines accélérateurs AMD). INT8 est l’approche classique par entiers, supportée depuis longtemps dans les bibliothèques quantization. Les deux visent le même but (réduire la précision à 8 bits), mais fonctionnent différemment : FP8 a une mantisse/exposant, ce qui lui donne une portée plus flexible (valeurs très petites ou très grandes) pour une même taille docs.nvidia.com , tandis que INT8 a une dynamique fixe mais aucune “magie” d’exposant (il faut bien choisir les échelles). W8A8 (weights and activations 8-bit) peut désigner l’un ou l’autre. Sur H100, 8-bit aura tendance à signifier FP8 car c’est ce qui donne le meilleur résultat. Sur A100 ou d’autres, 8-bit impliquera plutôt INT8 (SmoothQuant ou autre). Il est donc toujours bon de préciser.

Q4 – PagedAttention, c’est quoi déjà ? C’est la technologie de vLLM qui gère la mémoire du KV-cache en “pages” plutôt qu’en blocs contigus monolithiques. En divisant le KV-cache de chaque requête en petits segments, on peut les allouer et les libérer de manière flexible, un peu comme la mémoire virtuelle d’un OS blog.vllm.ai . Ainsi, on élimine la fragmentation interne/externe (chaque page non utilisée peut servir ailleurs) et on permet de partager des pages entre requêtes (notamment pour le prefix-batching ou le beam search où plusieurs générations partagent le même contexte initial) blog.vllm.ai . L’effet est un gaspillage mémoire quasi nul (<4%) et la possibilité de batcher énormément de requêtes sans exploser la VRAM. PagedAttention n’a pas d’impact sur la qualité du modèle (c’est transparent côté résultats), mais booste le throughput en permettant une meilleure utilisation du GPU blog.vllm.ai ar5iv.labs.arxiv.org . C’est vraiment une avancée clé pour servir les LLM à grande échelle.

Q5 – Et les quants GGUF (Q8_0, Q4_K_M, …) dont on voit les noms ? Ce sont les différents presets de quantization utilisés avec llama.cpp et d’autres outils CPU. En gros, Q8_0 signifie 8 bits non groupé (toutes les matrices quantifiées globalement, sans offset par groupe), c’est la version la plus fidèle (presque sans perte, on gagne surtout en taille mémoire). Q4_K_M signifie 4 bits, quantization groupée par blocs de 128 (K), niveau Medium (M) de précision : en pratique ça utilise des échelles séparées par groupes de neurones, ce qui améliore la fidélité par rapport à un simple 4-bit homogène. Il existe aussi Q4_0 (4-bit de base), Q4_K_S (4-bit grouped Small), Q5_0, Q5_K_M, etc. La qualité varie un peu en conséquence : Q8_0 est très proche du modèle original, Q4_K_M est l’un des meilleurs compromis en 4-bit, Q4_0 est plus hasardeux (surtout sur modèles >30B). Pour un GPU H100, ces formats ne tirent pas parti du hardware spécial (ils seront traités comme des INT8 en gros), donc on leur préférera FP8/INT8 via TRT-LLM ou vLLM. En revanche, pour un CPU ou un petit GPU, les quants GGUF sont super : ils permettent de tester rapidement un modèle sans mobiliser 80 Go de RAM. On peut par exemple lancer un Llama2 13B Q4_K_M sur un PC 16 Go RAM – la génération sera lente, mais ça fonctionne. Donc ces quantizations ont leur place dans l’écosystème, mais ce ne sont pas celles qu’on utilisera pour une prod optimale sur H100.

Cas particulier : modèles “merge” et licences
Il existe des modèles obtenus par merge (fusion de plusieurs checkpoints) qui posent des questions de licence. Par exemple Luminum-123B est un modèle 123 milliards résultant du merge de : Mistral-Large-Instruct-2407 (base), Lumimaid-v0.2-123B, et Magnum-v2-123B huggingface.co huggingface.co . Chacune de ces composantes a sa propre licence :

Lumimaid-123B (aussi appelé NeverSleep/Lumimaid-v0.2-123B) est en licence CC-BY-NC-4.0 (Creative Commons Attribution Non-Commercial) huggingface.co . Cela signifie usage non commercial uniquement, partage autorisé tant qu’on crédite l’auteur, pas de dérivés commerciaux.

Mistral-Large-Instruct-2407 est sous licence Mistral AI Research License (MRL) huggingface.co . C’est une licence propriétaire de Mistral AI qui autorise l’usage recherche et le self-hosting non commercial, mais interdit l’usage commercial sans accord explicite. Elle interdit également de distribuer les poids dérivés à des tiers sans passer par un accord avec Mistral AI huggingface.co huggingface.co . En gros, c’est non-commercial avec des restrictions supplémentaires (pas d’exploitation commerciale du modèle ni de ses dérivés sans licence payante).

Magnum 123B quant à lui (si on reprend l’exemple) a probablement une licence du même acabit (souvent les modèles « roleplay » sont en Llama2-Community ou autre, on va supposer non-commercial aussi).

En combinant ces modèles, Luminum hérite des restrictions les plus fortes de chacun. Autrement dit, Luminum-123B est non-commercial (à cause de Lumimaid CC-BY-NC et Mistral MRL) et ne peut pas être distribué librement en tant que poids merge sans accord (surtout à cause de Mistral MRL qui impose de ne pas partager de dérivés). Pour cette raison, l’auteur de Luminum a publié son modèle sur HuggingFace mais en marquant qu’il faut accepter la MRL pour y accéder, et en rappelant qu’il ne faut pas utiliser ça commercialement.

Conséquence pratique : si vous quantifiez un modèle issu d’un merge sous restriction non-commerciale, vous ne pouvez pas republier les poids quantifiés (même en GGUF ou engine TRT) en prétendant lever la restriction – la quantization ne change pas la licence du contenu. Il faut traiter cela comme un modèle original pour la licence. Donc, pas de distribution publique de Luminum quantifié sans autorisation. À la place, ce qu’on peut faire c’est partager des instructions de reproduction (par ex. un script de merge + quantization que chacun peut exécuter de son côté après avoir accepté les licences sources). On peut aussi éventuellement distribuer des delta weights ou LoRA si la licence le permet (par ex. Lumimaid étant open non-commercial, un LoRA dessus reste NC).

En somme, faites bien attention aux licences des modèles et de leurs données d’entraînement. Un modèle comme Llama2 70B base est Llama2-community (autorisation commerciale), mais sa version fine-tunée par X peut être Apache-2.0 ou NC, etc. Toujours vérifier sur la carte HuggingFace ! Dans le doute, abstenez-vous de diffuser un dérivé.

(Exemple réel : Luminum étant NC, un utilisateur ne doit pas l’utiliser dans un produit payant. S’il voulait une version commerciale, il devrait entraîner ou acquérir un modèle équivalent sous licence permissive. Mistral AI vend une licence pro pour son 7B instruct, par exemple.)

Choisir sa quantization (arbre de décision)
Pour clôturer, voici un petit guide décisionnel pour choisir le bon niveau de quantization selon vos besoins :

Qualité quasi FP16 + perfs maximales (GPU H100) : Optez pour le FP8 (W8A8). C’est idéal si vous avez des H100 ou MI300 récents : vous obtiendrez le meilleur débit et des réponses presque identiques à FP16. Stacks conseillées : TensorRT-LLM si vous visez les toutes meilleures latences et un déploiement C++ optimisé developer.nvidia.com , ou vLLM en FP8 si vous voulez rester en full Python OSS developers.redhat.com . Dans les deux cas, activez le KV-cache en FP8 pour bénéficier de la mémoire gagnée.

8-bit “classique” toutes plateformes : INT8 SmoothQuant (W8A8). Si vos GPU ne supportent pas FP8 (ex : A100) ou si vous tenez à une solution éprouvée, le combo poids+act en INT8 calibré est un excellent choix. SmoothQuant a fait ses preuves sur LLM >100B sans perte significative arxiv.org . Stacks conseillées : vLLM --quantization int8, ou des runtimes comme FasterTransformer sur A100 (int8 sans FP8). N’oubliez pas que INT8 fonctionne aussi bien sur CPU (on commence à voir des accélérations int8 sur CPU via ONNXRuntime par ex).

Compression agressive / VRAM limitée : INT4 (AWQ/GPTQ). Si vous devez faire tenir un modèle très gros dans peu de mémoire, ou lancer plein d’instances parallèles, le 4-bit weight-only est la solution. Vous sacrifierez un peu de “humanité” dans les réponses (phrases un peu plus génériques, style moins raffiné), mais le modèle restera fonctionnel pour de nombreuses tâches. Stack conseillée : llama.cpp GGUF Q4_K_M ou AutoGPTQ (pour avoir un modèle 4-bit utilisable dans Transformers sur GPU). Sur H100, vous pouvez aussi combiner un modèle 4-bit avec un KV-cache FP8 via TensorRT-LLM (ils ont montré Falcon-180B en INT4 AWQ tournant sur un seul H200 dans un de leurs blogs!).

Prototypage rapide / Edge : GGUF (Q8_0, Q4_K, etc.) via llama.cpp. Si votre but est de tester un modèle en local, ou de le déployer sur une machine sans GPU NVIDIA, partez sur les quantizations fournies par la communauté en GGUF. Ça évite tout tracas d’installation et ça marche out of the box. La qualité dépend du preset (prendre de préférence les versions “K_M” en 4/5 bits pour un bon équilibre qualité). N’espérez pas la même vitesse qu’avec un GPU pro, mais pour des démos ou du dev c’est suffisant.

(En cas de doute, commencez par du FP16 ou FP8, voyez si la latence/mémoire vous conviennent, puis descendez d’un cran si nécessaire. Mieux vaut une réponse un peu lente mais fiable, qu’un modèle compressé à outrance mais décevant.)

Commandes types (référence rapide)
Voici un récapitulatif de quelques commandes évoquées, pour référence :

A) TensorRT-LLM (H100, FP8) – Exporter un modèle HF et builder en FP8 :

Export HF -> TRT-LLM checkpoint
python examples/llama/convert_checkpoint.py
--model_dir /chemin/vers/modele_hf
--output_dir /chemin/vers/output_trtllm_ckpt
--dtype float16 --tp_size 2 # si multi-GPU

Build engine FP8 + KV FP8
trtllm-build
--checkpoint_dir /chemin/vers/output_trtllm_ckpt
--output_dir /chemin/vers/engine_fp8
--use_fp8 --use_fp8_kv_cache
--max_batch_size 8
--max_input_len 8192 --max_output_len 1024
--tp_size 2 # si multi-GPU

Serveur TRT-LLM
trtllm-serve --engine_dir /chemin/vers/engine_fp8 --port 8080

(Cf. docs TRT-LLM pour plus de détails developer.nvidia.com . Pensez à ajuster batch_size et lengths à vos besoins réels pour optimiser la compilation.)

B) vLLM FP8 (W8A8 + KV FP8) – Lancer un serveur vLLM quantifié 8-bit :

vllm serve ORGANISATION/MODELE-HF
--quantization fp8
--kv-cache-dtype fp8
--max-model-len 16384

(Nécessite GPU H100 ou matériel supportant FP8. Cf. vLLM docs developers.redhat.com .)

C) vLLM INT8 (W8A8 + KV FP8) – Lancer vLLM en SmoothQuant 8-bit :

vllm serve ORGANISATION/MODELE-HF
--quantization int8
--kv-cache-dtype fp8
--max-model-len 16384

(Fonctionne sur A100/H100. Si pas de FP8 du tout, mettre kv-cache-dtype à fp16. On peut aussi charger un modèle AWQ en passant --quantization awq et en pointant vers le fichier .pt quantifié.)

D) Conversion GGUF (llama.cpp) – Convertir et quantifier un modèle en GGUF :

1. Convertir un modèle HF en GGUF FP16
python convert-hf-to-gguf.py NomDuModeleHF --outfile modele.gguf

2. Quantifier en 4 bits par ex.
./quantize modele.gguf modele-q4_0.gguf q4_0

(Voir documentation Qwen/llama.cpp qwen.readthedocs.io qwen.readthedocs.io . Il existe aussi des scripts pour appliquer AWQ avant conversion afin d’améliorer la qualité comme vu plus haut.)

Points de contrôle (qualité)
Avant de déployer en production, pensez à passer votre modèle quantifié par quelques points de contrôle qualité :

Jeu de validation : Préparez un set de prompts variés (10-50, selon vos ressources), couvrant les cas d’usage typiques. Idéalement multi-langues si concerné. Incluez des conversations multi-tours, des questions pièges, des demandes de génération créative, etc. Faites générer le modèle FP16 et le modèle quantifié sur ces prompts, et comparez. Cherchez les différences flagrantes (répétitions, ignorances d’instructions, réponses à côté…).

Métriques auto : Si possible, évaluez la perplexité du modèle quantifié sur un corpus de test. Un écart de perplexité très faible (<5-10%) par rapport au FP16 est bon signe. Vous pouvez aussi calculer des métriques de diversité lexicale comme distinct-n sur des longues générations : un modèle quantifié de façon agressive a parfois tendance à recycler les mêmes tournures, ce qui réduit distinct-4/5. Enfin, si votre application craint les hallucinations ou les refus injustifiés, testez-en quelques-uns (ex : demandes factuelles pour voir si le quant hallucine plus ; requêtes sensibles pour voir s’il se met à refuser inutilement).

A/B testing : Le mieux reste de faire évaluer quelques paires de réponses (FP16 vs quant) par des humains sans leur dire qui est qui. S’ils n’y voient que du feu ou préfèrent même parfois la version quantifiée, c’est gagné 🙂.

Réglage des hyperparamètres : Un modèle quantifié peut nécessiter de légers ajustements de sampling. En particulier, augmenter le repetition_penalty (p.ex. de 1.1 à 1.15) peut aider à garder le style cohérent sur de longues réponses. On peut aussi ajuster le top_p ou temperature si on constate des sorties moins variées. N’hésitez pas à tuner ces paramètres sur votre set de validation. Parfois, un quant de 4-bit appréciera une température un poil plus élevée pour compenser la perte de finesse.

Long contexte : Si vous visez du 16k ou 32k tokens, testez-le ! Envoyez un prompt de ~15k tokens et voyez si le modèle continue correctement. Sur de très longs contextes, la quantization peut accumuler de l’erreur numérique (d’où l’intérêt du KV en FP8 ou FP16). Assurez-vous que la dégradation reste gérable (de toute façon, au-delà de 8k même un modèle FP16 commence à flancher parfois).

En suivant ces points de contrôle, vous aurez l’assurance que votre modèle quantifié tient la route. La quantization est un art subtil : 99% du temps ça marche très bien, mais il vaut mieux débusquer le 1% de cas où ça pourrait poser souci avant que les utilisateurs ne tombent dessus.

TL;DR
H100 = FP8 natif 📈 : Les GPU NVIDIA Hopper (H100) supportent nativement le calcul en float8 via la Transformer Engine. Cela permet d’atteindre des performances jusqu’à ~4–5× supérieures à A100 FP16, avec une qualité de modèle pratiquement inchangée si calibré correctement nvidia.github.io developers.redhat.com . En clair, FP8 sur H100 offre le meilleur ratio qualité/latence/VRAM aujourd’hui.

TensorRT-LLM 🚀 : C’est la solution NVIDIA optimisée pour inférence LLM. Elle compile le modèle en un engine ultra-rapide. Avantages : support du FP8 et INT8 (SmoothQuant) directement, batching asynchrone en vol, KV-cache paginé, multi-GPU… Bref, c’est ce qui donnera les latences et throughputs minimum sur H100 developer.nvidia.com . Inconvénient : spécifique NVIDIA, et nécessite de passer par une étape de build.

vLLM 🐍 : Serveur haute performance open-source. Il introduit PagedAttention qui réduit le gâchis mémoire du KV-cache à <4%, permettant de booster le throughput sans changer de hardware runpod.io . vLLM supporte aussi FP8 et INT8 (ainsi que chargement de modèles 4-bit). Idéal si on veut une intégration simple (quelques lignes Python) tout en gardant des perfs state-of-the-art. C’est open-source (Apache 2.0). Moins rapide que TRT-LLM sur un seul GPU, mais plus flexible.

Choix de quantization 🤖 :

Pour la qualité max : FP8 (8-bit flottant) si possible, sinon INT8 SmoothQuant. Ces deux options donnent des résultats quasi identiques au FP16 original sur la plupart des modèles arxiv.org .

Pour pousser la compression : INT4 (4-bit poids) via AWQ/GPTQ est faisable sur des grands modèles, au prix d’une très légère dégradation du style/cohérence. À utiliser si VRAM limitée ou pour héberger plusieurs instances.

Le tout sans réentraîner (PTQ). On peut quantizer un modèle après-coup et le servir directement.

Formats GGUF (llama.cpp) 💾 : Utiles pour exécuter des LLM sur CPU ou petits GPU. Exemples : Q8_0 (8-bit poids), Q4_K_M (4-bit groupe Medium) medium.com . Ils rendent les modèles plus accessibles, au prix d’une vitesse moindre. Sur H100, ces formats ne tirent pas profit du hardware spécialisé, donc on privilégiera plutôt TRT-LLM/vLLM. Mais pour du offline ou du local sans CUDA, c’est génial.

Licences & modèles merges 📜 : Attention à la légalité ! Un modèle comme Luminum-123B mergeant Mistral (licence MRL, non-commercial) et Lumimaid (CC-BY-NC-4.0) reste Non-Commercial et soumis aux restrictions de diffusion des originaux huggingface.co huggingface.co . Quantizer un modèle ne change pas sa licence. Il est généralement interdit de redistribuer des poids dérivés sans accord si la licence source l’interdit (ex : Mistral MRL prohibe de partager le modèle fine-tuné sans passer par eux huggingface.co ). Préférez partager des scripts ou des diffs/LoRA plutôt que les poids quantifiés directement pour ces cas. En clair : toujours respecter les licences, même pour un modèle quantifié ou compressé !

0 commit comments
Comments
0
 (0)
Comment
You're not receiving notifications from this thread.

Update Article.md · SoMika00/Quant_llm@c4335f0 There are no files selected for viewing





debut de la reecriture : Guide pratique (2025) — Quantization LLM sur H100 et alternatives (TRT-LLM, vLLM, GGUF)
Introduction : Les bases de la mémoire d'un LLM
Pour bien comprendre la quantization, il est essentiel de maîtriser quatre concepts clés qui déterminent l'empreinte mémoire d'un grand modèle de langage (LLM) lors de l'inférence : les poids, les activations, le KV-cache et les formats numériques.
Poids et Activations : Un tenseur est un tableau multi-dimensionnel de nombres. Dans un LLM, les poids (les paramètres du modèle appris durant l'entraînement) sont stockés sur disque puis chargés en mémoire vive (VRAM du GPU) sous forme de tenseurs. Les activations désignent les résultats intermédiaires calculés à chaque étape de l'inférence.
KV-Cache : La mémoire KV-cache correspond aux tenseurs de clés (Keys) et valeurs (Values) du mécanisme d'attention, conservés au fil de la génération de texte. Ce cache est crucial car il accélère l'auto-régression en évitant de recalculer tout l'historique à chaque nouveau token généré. Sa gestion est optimisée sur les serveurs modernes, notamment grâce à des techniques comme Paged Attention (qui partitionne le cache en pages) et l'in-flight batching (qui regroupe des requêtes en cours). La bibliothèque open-source vLLM, par exemple, utilise PagedAttention pour stocker les Keys/Values en blocs non contigus, réduisant le gaspillage de mémoire à moins de 4 % (contre 60 à 80 % dans les systèmes classiques).
Répartition de la VRAM : En pratique, la VRAM requise à l’inférence se compose des poids du modèle, des activations temporaires, et du KV-cache. Les poids dominent souvent l'empreinte mémoire (environ 65 % de la VRAM sur un modèle 13B), le KV-cache occupant ~30 % (variable selon la longueur de séquence) et les activations une part minime. La taille sur disque est donc dictée par les poids, tandis que l'usage en VRAM dépend fortement du KV-cache pour les longs contextes.
Overflow / Underflow : Un overflow survient quand une valeur dépasse la plage représentable par le format numérique choisi (elle devient infinie), tandis qu’un underflow se produit lorsque la valeur est trop petite pour être représentée (elle est arrondie à zéro). Les formats à faible dynamique, comme FP8 ou INT4/INT8, nécessitent des techniques de calibration pour éviter ces problèmes. La méthode SmoothQuant, par exemple, ajuste l'échelle des poids et des activations pour "lisser" les valeurs extrêmes (outliers) avant la quantization.
Les formats numériques : de la haute précision à l'ultra-compression
Le choix du format numérique est un compromis permanent entre précision, performance et consommation mémoire.
FP32 (float 32 bits) : Le format de référence en entraînement. Il offre une haute précision et une large plage dynamique (~10^38), au prix d'un coût mémoire et calcul élevé.
BF16 (bfloat16) : Format 16 bits avec la même plage dynamique que le FP32 mais une précision réduite. Très utilisé en entraînement en précision mixte sur TPU/GPU car il préserve bien l'échelle des gradients.
FP16 (float16) : Format 16 bits standard en inférence GPU. Sa plage de valeurs est plus réduite que le BF16, mais il offre une meilleure précision. Les Tensor Cores des GPU Ampere/Hopper y sont dédiés.
FP8 (float8) : Introduit avec l'architecture NVIDIA Hopper (H100), il existe en deux variantes : E4M3 (plus de précision, plage modérée) et E5M2 (moins de précision, plage plus large). Le H100 l'accélère nativement via sa Transformer Engine, permettant de diviser par deux l'empreinte mémoire par rapport au FP16 avec une qualité quasi-identique si le modèle est bien calibré.
INT8 (entier 8 bits) : La quantization en entiers 8 bits (valeurs de -128 à 127) nécessite des échelles (scales) pour mapper la plage de valeurs réelles aux 256 niveaux disponibles. On distingue la quantization des poids seuls de la quantization W8A8 (poids et activations), qui permet d'accélérer l'ensemble du calcul grâce aux Tensor Cores INT8. La méthode SmoothQuant a démontré l'efficacité du W8A8 en déplaçant la difficulté de quantization des activations vers les poids.
INT4 (entier 4 bits) : Une quantization ultra-agressive (16 niveaux) appliquée uniquement aux poids (weight-only). Elle est utilisée pour compresser massivement les modèles sur du matériel contraint (GPU grand public, CPU). Des techniques comme AWQ ou GPTQ parviennent à réduire la perte de qualité en utilisant des échelles par groupe de poids ou en protégeant les poids les plus importants. Le gain de vitesse n'est pas aussi élevé que le gain mémoire, car l'INT4 n'est pas accéléré nativement.
Tableau récapitulatif des formats
Format	Bits (total)	Exposant/Mantisse	Plage dynamique approx.	Utilisation typique et remarques
FP32	32 bits	8 exp, 23 mant	~1e-38 à 1e+38	Haute précision (référence). Entraînement, calculs sensibles.
BF16	16 bits	8 exp, 7 mant	~1e-38 à 1e+38	Entraînement mixte précision (TPU/GPU), inférence. Même dynamique que FP32 mais précision réduite.
FP16	16 bits	5 exp, 10 mant	~1e-4 à 6.5e+4	Inférence sur GPU (Tensor Cores). Précision suffisante dans la plupart des cas.
FP8 E4M3	8 bits	4 exp, 3 mant	~1e-2 à ~4.5e+2	Inférence GPU Hopper (H100). Utilisé pour poids/activations. Calibration impérative.
FP8 E5M2	8 bits	5 exp, 2 mant	~1e-2 à ~5.7e+4	Utilisé pour les gradients (plus grande dynamique) ou le KV-cache FP8.
INT8 (W8A8)	8 bits (entier)	256 valeurs (échelle)	Inférence quantifiée poids + activations. Requiert calibration (ex. SmoothQuant).	
INT4 (poids)	4 bits (entier)	16 valeurs (par groupe)	Compression extrême des poids. Légère dégradation possible. Utilisé via AWQ, GPTQ.	
Objectifs et stratégies de quantization par défaut
La quantization n'est pas une fin en soi, mais un moyen d'atteindre des objectifs précis de déploiement. Voici trois stratégies par défaut et leurs cas d'usage.
H100 FP8 de bout en bout (Qualité maximale, performance extrême)
Objectif : Obtenir la latence la plus faible et le débit le plus élevé possible sur du matériel de dernière génération (NVIDIA H100), sans sacrifier la qualité du modèle.
Description : Utilisation du format FP8 pour les poids, les activations et le KV-cache. Cette approche tire pleinement parti de l'architecture Hopper et de sa Transformer Engine. La perte de qualité est généralement négligeable (<1 %) par rapport au FP16, tandis que les performances sont multipliées.
Quand ne pas l'utiliser : Si vous ne disposez pas de GPU compatibles (Hopper, Ada ou futurs), ou si des tests révèlent une instabilité numérique sur des couches très spécifiques de votre modèle (un cas rare).
INT8 SmoothQuant (Le choix robuste et universel)
Objectif : Réduire de moitié l'empreinte mémoire et accélérer les calculs sur une large gamme de matériel (GPU Ampere/Hopper, CPU), avec une garantie de stabilité et une perte de qualité minimale.
Description : Quantization des poids et des activations en entiers 8 bits (W8A8) grâce à la méthode SmoothQuant, qui calibre les échelles pour éviter les erreurs. C'est une solution éprouvée, bien supportée par les frameworks.
Quand ne pas l'utiliser : Sur H100, le FP8 est souvent légèrement plus performant. Si la moindre dégradation de la qualité est inacceptable pour des tâches de raisonnement très complexes (bien que la perte soit quasi nulle).
INT4 AWQ + KV-Cache FP8 (Compression maximale, haute densité)
Objectif : Maximiser le nombre de modèles ou de sessions simultanées sur une VRAM donnée, ou déployer des modèles très larges sur du matériel contraint.
Description : Les poids du modèle sont compressés en 4 bits avec une méthode avancée comme AWQ (Activation-aware Weight Quantization) pour préserver la qualité. Les activations restent en FP16 et le KV-cache peut être stocké en FP8 pour économiser encore plus de mémoire.
Quand ne pas l'utiliser : Pour des applications où la finesse stylistique, la cohérence sur de très longs textes ou le raisonnement complexe sont critiques. La perte de qualité, bien que modeste, est plus perceptible qu'en 8 bits.
Tableau décisionnel : Quand choisir quoi ?
Ce tableau vous aide à choisir la bonne stratégie en fonction de vos contraintes.
Contrainte	FP8 (W8A8)	INT8 SmoothQuant (W8A8)	INT4 AWQ (poids)	GGUF (CPU/Edge)
Latence minimale	✅ (H100)	✅ (GPU récents)	⚠️ (Gain mémoire > vitesse)	❌ (Lent)
VRAM maximale	✅	✅	✅✅ (Optimal)	✅ (Très faible conso)
Qualité max (proche FP16)	✅✅	✅	⚠️ (Légère dégradation)	⚠️ (Dépend du format)
Portabilité (CPU/GPU)	❌ (H100+)	⚠️ (GPU/CPU avec support)	⚠️ (GPU/CPU avec support)	✅✅ (Universel)
Flag vLLM	--quantization fp8	--quantization int8	--quantization awq	(Chargement direct du .gguf)
Option TensorRT-LLM	--use_fp8	--quant_mode int8_sq	--quant_mode int4_awq	(Non applicable)
La gestion du KV-cache : un enjeu majeur
L'impact du format : FP16 vs FP8
Par défaut, le KV-cache est souvent maintenu en FP16 pour une fidélité maximale, mais cela consomme beaucoup de VRAM. Passer le KV-cache en FP8 divise par deux cette empreinte, permettant d'augmenter la longueur de contexte ou le nombre de sessions simultanées. En pratique, l'impact sur la qualité est quasi nul avec un FP8 bien calibré sur H100.
La formule du KV-Cache (avec GQA)
La taille en octets du KV-cache peut être estimée avec la formule suivante :
KV_bytes ≈ batch_size × sequence_length × num_layers × (2 × hidden_size) × bytes_per_dtype / gqa_factor
gqa_factor : Ratio entre le nombre de têtes d'attention et le nombre de têtes de clés/valeurs (pour Grouped-Query Attention). Vaut 1 pour MHA, >1 pour GQA/MQA.
bytes_per_dtype :
Data Type	Bytes
FP32	4
FP16 / BF16	2
FP8 / INT8	1
Conclusion : Passer le KV-cache en FP8 permet de servir environ deux fois plus de sessions ou de gérer un contexte deux fois plus long à VRAM constante.
L'optimisation structurelle : PagedAttention
Indépendamment du format, des algorithmes comme PagedAttention (popularisé par vLLM) optimisent l'usage du KV-cache en le découpant en pages. Cela évite la fragmentation mémoire et permet de partager des pages entre requêtes, réduisant drastiquement le gaspillage. Combiner KV-cache FP8 et PagedAttention offre le meilleur des deux mondes : un cache compact et géré efficacement.
Méthodes de quantization clés
SmoothQuant (INT8 W8A8) : Lisse les outliers d'activation en transférant une partie de leur amplitude vers les poids via un simple rescaling. C'est une méthode de post-training (sans ré-entraînement) qui permet de quantifier des modèles jusqu'à 530 milliards de paramètres en 8 bits avec une perte de précision négligeable.
AWQ (Activation-aware Weight Quantization, INT4) : Identifie et protège les ~1% de poids les plus critiques pour la performance (identifiés via l'analyse des activations) et quantifie agressivement les 99% restants en 4 bits. Cette méthode a reçu le prix du meilleur article à MLSys 2024.
GPTQ (INT3/INT4) : Utilise des informations de second ordre (approximation Hessienne) pour minimiser l'erreur de quantization bloc par bloc. C'est une méthode one-shot rapide et efficace qui a été largement adoptée par la communauté open-source.
LLM.int8() (bitsandbytes) : Une approche qui quantifie la majorité des opérations en INT8 tout en isolant les outliers dans une multiplication séparée en FP16. Elle réduit l'empreinte mémoire sans accélérer le calcul et est aujourd'hui souvent remplacée par des solutions W8A8 plus performantes sur H100.
Piles logicielles : TRT-LLM vs vLLM vs llama.cpp/GGUF
A) TensorRT-LLM (NVIDIA)
Un runtime et compilateur optimisé par NVIDIA. Il prend un modèle et le compile en un moteur binaire ultra-performant, spécifique au GPU.
Points forts : Support FP8 natif sur H100, INT8/INT4, in-flight batching, paged KV-cache, et multi-GPU. Offre les meilleures performances brutes (latence et débit).
Limites : Spécifique à NVIDIA, moins flexible (la compilation est nécessaire), et peut avoir un délai de support pour les nouveaux modèles exotiques.
B) vLLM (Open-Source)
Un serveur d'inférence LLM développé par UC Berkeley, axé sur la performance et la flexibilité.
Points forts : Intègre l'algorithme PagedAttention pour une gestion mémoire quasi-optimale. Supporte FP8, INT8, et le chargement de poids 4-bit (AWQ, GPTQ). API Python simple, batching dynamique et open-source (Apache 2.0).
Limites : Légèrement moins performant que TRT-LLM en termes de latence brute sur une seule requête, mais souvent plus efficace en débit sous forte charge multi-utilisateurs grâce à PagedAttention.
C) llama.cpp / GGUF (CPU & autres)
Un écosystème C++ pour exécuter des LLM sur du matériel varié (CPU, Apple Silicon). GGUF est le format de fichier utilisé pour stocker les poids quantifiés.
Points forts : Multiplateforme, facile à déployer, large communauté. Supporte de nombreux formats de quantization spécialisés (Q4_K_M, Q8_0, etc.). Idéal pour le prototypage local et le déploiement sur des machines sans GPU puissant.
Limites : Non optimisé pour les Tensor Cores des GPU haut de gamme. Utiliser llama.cpp sur un H100 serait un énorme gaspillage de potentiel.
Calibration et garde-fous pour FP8/INT8 : une checklist pratique
Une quantization réussie n'est pas automatique. Voici une checklist pour assurer la robustesse.
Jeu de calibration : Utilisez un jeu de données représentatif de votre cas d'usage en production (512 à 2048 prompts suffisent généralement). La qualité des données de calibration influe directement sur la performance du modèle quantifié.
Exclusion de couches sensibles : Si vous observez une instabilité, excluez les couches d'embedding et la tête de prédiction du langage (lm_head) de la quantization. Ces couches sont souvent plus sensibles à la perte de précision.
Recette de quantization :
Scaling : Privilégiez un scaling per-tensor ou per-channel pour une granularité fine.
FP8 : Appliquez une recette mixte. Utilisez E4M3 pour les couches nécessitant de la précision (comme les matrices de poids) et E5M2 pour celles nécessitant une plus grande dynamique (comme les gradients, ou parfois les activations avec de grands outliers).
Contrôles de validité (Sanity Checks) :
Mesurez le taux de saturation et le nombre d'overflows pendant la calibration. Des valeurs élevées indiquent un problème.
Calculez la similarité cosinus entre les sorties du modèle FP16 et du modèle quantifié. Un score élevé (>0.99) est un bon indicateur.
Stratégie de repli (Fallback) : En cas de doute, surtout pour des tâches de raisonnement complexe, conservez le KV-cache en FP16 (--kv-cache-dtype fp16). C'est un excellent compromis qui préserve la fidélité des attentions tout en bénéficiant de la quantization des poids/activations.
Observabilité en production (SLOs)
Déployer un modèle quantifié, c'est bien. Le surveiller, c'est mieux.
Indicateurs clés à suivre :
Latence : p99 du temps de génération du premier token.
Débit : tokens/seconde.
Utilisation VRAM : VRAM par requête, fragmentation du KV-cache.
Erreurs : Taux d'erreurs Out-Of-Memory (OOM), taux d'abandon des requêtes.
Dérive de la qualité : Perplexité calculée de manière hebdomadaire sur un jeu de données de contrôle.
Alertes à configurer :
Latence p99 > seuil défini.
Utilisation VRAM > 95%.
Taux OOM > 0.5% des requêtes.
Gaspillage du KV-cache > 8%.
Risques, cas limites et comment les gérer
Saturation du FP8 E4M3 : Sur des couches à très grande dynamique, le format E4M3 peut saturer. La solution est de forcer localement l'utilisation du format E5M2 pour ces couches spécifiques.
Long contexte et raisonnement complexe : Pour des tâches très sensibles qui s'étalent sur des dizaines de milliers de tokens, la légère perte de précision du KV-cache FP8 peut s'accumuler. Testez rigoureusement avec un KV-cache FP16 pour valider si la différence est significative pour votre cas d'usage.
"Perte de style" en INT4 : La quantization agressive en 4 bits peut parfois rendre le style du modèle plus générique. On peut souvent compenser ce phénomène en ajustant légèrement les paramètres de génération : augmenter un peu repetition_penalty (ex: de 1.1 à 1.15) et temperature.
Licences : la quantization ne change rien
La quantization est une transformation technique, elle ne modifie en rien la licence d'un modèle.
Modèle (Exemple)	Licence	Usage Commercial	Redistribution Poids Quantifiés	Stratégie de Partage Recommandée
Llama 3 70B	Llama 3 Community License	✅	✅	Poids directs (GGUF, AWQ, etc.)
Mistral Large	Mistral AI Research License (MRL)	❌	❌	Scripts de quantization/merge
Luminum-123B (Merge)	Hérite de la plus restrictive (MRL, CC-BY-NC)	❌	❌	Scripts de merge + quantization
Gemma 7B	Gemma Terms of Use	✅	✅	Poids directs
Règle d'or : Un modèle issu d'un merge hérite des restrictions de toutes ses composantes. En cas de doute, partagez des scripts de reproduction (merge, fine-tuning, quantization) plutôt que les poids dérivés.
Annexe : Commandes types (Cheat-Sheet)
A) TensorRT-LLM (H100, FP8)
Exporter le checkpoint HF
code
Bash
python examples/llama/convert_checkpoint.py \
  --model_dir /path/to/model_hf \
  --output_dir /path/to/output_trtllm_ckpt \
  --dtype float16 --tp_size 2
Builder l'engine FP8
code
Bash
trtllm-build \
  --checkpoint_dir /path/to/output_trtllm_ckpt \
  --output_dir /path/to/engine_fp8 \
  --use_fp8 --use_fp8_kv_cache \
  --max_batch_size 8 \
  --max_input_len 8192 --max_output_len 1024 \
  --tp_size 2
Lancer le serveur
code
Bash
trtllm-serve --engine_dir /path/to/engine_fp8 --port 8080
B) vLLM (FP8 ou INT8)
Lancer en FP8
code
Bash
vllm serve ORG/MODEL-HF \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
Lancer en INT8 (SmoothQuant)
code
Bash
vllm serve ORG/MODEL-HF \
  --quantization int8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384
C) Conversion GGUF (llama.cpp)
Convertir HF en GGUF FP16
code
Bash
python convert-hf-to-gguf.py ModelNameHF --outfile model.gguf
Quantifier en 4 bits
code
Bash
./quantize model.gguf model-q4_K_M.gguf q4_K_M
