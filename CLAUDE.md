# smthsmth — Classification d'actions vidéo (32 classes)

## Tâche

Classification supervisée d'actions vidéo sur un sous-ensemble du dataset
**Something-Something**. Pour chaque vidéo, prédire l'une des **32 classes
d'action** (manipulation d'objets : ouvrir, fermer, pousser, verser, plier,
faire semblant de…, etc.).

Chaque vidéo est représentée sous deux modalités complémentaires :

- **frames/** — images RGB extraites de la vidéo (apparence)
- **optical_flow/** — flux optique TVL1 entre frames consécutives (mouvement)

L'idée pédagogique typique de ce setup : entraîner un modèle **two-stream**
(une branche RGB, une branche flux), ou un 3D-CNN, et comparer.

## Classes (32, indexées 000–032 avec 027 manquant)

```
000 Closing_something                       017 Pretending_to_throw_something
001 Covering_something_with_something       018 Pulling_something_from_left_to_right
002 Dropping_something_into_something       019 Pulling_something_from_right_to_left
003 Folding_something                       020 Putting_something_behind_something
004 Hitting_something_with_something        021 Putting_something_in_front_of_something
005 Holding_something                       022 Putting_something_into_something
006 Moving_something_away_from_something    023 Putting_something_next_to_something
007 Moving_something_closer_to_something    024 Putting_something_onto_something
008 Moving_something_down                   025 Showing_something_to_the_camera
009 Moving_something_up                     026 Spilling_something_next_to_something
010 Opening_something                       028 Taking_something_out_of_something
011 Picking_something_up                    029 Throwing_something
012 Pouring_something_into_something        030 Turning_something_upside_down
013 Pouring_something_out_of_something      031 Uncovering_something
014 Pretending_to_pick_something_up         032 Unfolding_something
015 Pretending_to_pour_something_..._but_   (027 absent — le label "32" couvre 32 classes)
016 Pretending_to_put_something_into_something
```

Attention : les indices vont jusqu'à 032 mais **027 est absent** → 32 classes
et non 33. Pour la sortie d'un classifieur on remappera vers `[0..31]`.

## Arborescence

```
smthsmth/
├── frames/
│   ├── train/   ← 32 dossiers de classe, 44 992 vidéos au total
│   │   └── 000_Closing_something/
│   │       └── video_10061/
│   │           ├── frame_000.jpg
│   │           ├── frame_001.jpg
│   │           ├── frame_002.jpg
│   │           └── frame_003.jpg
│   ├── val/     ← 32 dossiers de classe, 6 740 vidéos
│   └── test/    ← PAS de dossier de classe — 6 909 vidéos "à plat"
│       └── video_1000/   (frames brutes, label inconnu = à prédire)
│
├── optical_flow/   (mêmes splits, mêmes vidéos, structure miroir)
│   └── train/000_Closing_something/video_10061/
│       ├── flow_x_000.jpg   ┐
│       ├── flow_y_000.jpg   │  paire (frame_000 → frame_001)
│       ├── flow_x_001.jpg   ┐
│       ├── flow_y_001.jpg   │  paire (frame_001 → frame_002)
│       ├── flow_x_002.jpg   ┐
│       └── flow_y_002.jpg   │  paire (frame_002 → frame_003)
│
├── archives/                 ← format de travail réel (voir section dédiée)
│   ├── frames_train.tar  + frames_train.tar.index.json
│   ├── frames_val.tar    + frames_val.tar.index.json
│   ├── frames_test.tar   + frames_test.tar.index.json
│   ├── flow_train.tar    + flow_train.tar.index.json
│   ├── flow_val.tar      + flow_val.tar.index.json
│   └── flow_test.tar     + flow_test.tar.index.json
│
├── compute_optical_flow.py   ← génère optical_flow/ depuis frames/
├── pack_dataset.py           ← empaquette frames/ + optical_flow/ → archives/
├── clean_dataset.py          ← purge la pollution flow et les vidéos incomplètes
├── main.py                   ← entraînement, lit UNIQUEMENT archives/
├── debug_train.py            ← smoke-test pour tous les modèles de models_code/
├── report_run.py             ← récap d'un run (texte + 3 figures PNG)
├── models_code/              ← un fichier = un modèle (cf. section dédiée)
│   ├── CNN_rgb.py            ← baseline 2D, première frame uniquement
│   ├── CNN_flow.py           ← baseline 2D, 6 cartes flow empilées
│   ├── CNN_two_stream.py     ← two-stream CNN partagé, fusion concat features
│   ├── TSM_two_stream.py     ← Temporal Shift Module sur les deux modalités
│   └── R2plus1D_two_stream.py← (2+1)D : conv spatiale puis temporelle séparées
├── requirements.txt
├── runs/                     ← un sous-dossier par entraînement (jamais écrasé)
│   └── <model>_<YYYYMMDD-HHMMSS>/
│       ├── config.json       (args, sysinfo, git, nb params, taille dataset)
│       ├── history.csv       (1 ligne par epoch : losses, top1, top5, lr, durée)
│       ├── summary.json      (best/final acc, durée totale, per-class, conf. matrix)
│       ├── best.pth          (state_dict + args + epoch + val_acc)
│       └── report_*.png      (curves, per_class, confusion — générés en fin de run)
└── .venv/
```

### Détails clés

- **4 frames par vidéo** (frame_000 à frame_003) → 3 paires de flux par vidéo
  (flow_x/flow_y indices 000, 001, 002).
- **Split test sans labels** : les vidéos sont à la racine de `test/` (pas de
  dossier de classe). C'est le set d'évaluation finale — il faudra produire
  un fichier de prédictions (`video_id, predicted_class`).
- **train/val ont la structure label/** : facile pour `ImageFolder`-like ou
  un `Dataset` PyTorch custom.
- **Tailles** (post-cleanup, cf. `clean_dataset.py`) :
  - train = 44 992 vidéos (déséquilibré : ex. classe 000 a ~1 068, classe
    032 en a ~840)
  - val = 6 740 vidéos
  - test = 6 909 vidéos

## Le script `compute_optical_flow.py`

- Calcule le **flux optique TVL1** (OpenCV `cv2.optflow.DualTVL1OpticalFlow`,
  d'où `opencv-contrib-python` dans les requirements).
- Pipeline : pour chaque paire de frames consécutives, calcule (flow_x, flow_y)
  en float, **clip à [-20, 20]**, puis normalise à uint8 [0, 255] et écrit en
  JPG.
- Multi-process via `ProcessPoolExecutor` (param `--workers`, défaut 4).
- Re-exécutable : reproduit la structure `frames/{split}/{class}/{video}/`
  sous `optical_flow/`.
- Déjà exécuté — `optical_flow/` est rempli.

## Archives tar — format de travail réel

`frames/` et `optical_flow/` représentent ~**587 000 petits fichiers JPG**
(235k frames + 352k flows) pour ~5,8 Go. Cette explosion en nombre de fichiers
rend les transferts (rsync/scp) et certaines I/O très lentes. On empaquette donc
le dataset en **6 archives tar non compressées** — c'est le format que main.py
et debug_train.py lisent désormais. **Les dossiers d'origine ne sont plus lus
par le pipeline d'entraînement.**

### Format

Pour chaque (split, modalité), un tar + un index JSON :

```
archives/frames_train.tar              archives/frames_train.tar.index.json
archives/flow_train.tar                archives/flow_train.tar.index.json
... (idem val, test)
```

- **Tar non compressé** : les JPG sont déjà compressés, gzip n'apporte rien et
  casserait l'accès aléatoire.
- **Index JSON** : `{nom_membre: [offset_data, size]}`. Permet un `seek+read`
  direct dans le tar sans passer par `tarfile.extract*()`.
- Structure interne du tar :
  - train/val : `<class_dir>/<video_dir>/<file>.jpg` (ex.
    `000_Closing_something/video_10061/frame_000.jpg`)
  - test : `<video_dir>/<file>.jpg` (pas de classe — labels inconnus)

### Génération (`pack_dataset.py`)

```bash
uv run pack_dataset.py
# Options : --frames-root frames --flow-root optical_flow --out archives
#           --splits train val test
```

Itère récursivement les fichiers, écrit le tar avec `tarfile.open(..., "w")`,
puis re-lit le tar pour récupérer les `offset_data` / `size` réels de chaque
membre et sauve l'index JSON.

### Lecture (`main.py` :: `TarReader`)

`TarReader` ouvre l'index JSON dans `__init__` et garde le file descriptor à
`None` ; le fd est ouvert paresseusement à la première lecture
(`read(name) -> bytes`). C'est **fork-safe** pour le DataLoader multi-workers :
tant que le main process ne lit rien, chaque worker fork ouvre son propre fd
après le fork (pas de seek partagé entre processus). `__getstate__` strip le fd
pour le mode "spawn".

`SmthSmthDataset` :
- `__init__(archives_root="archives", split, mode, image_size)` — ne reçoit
  plus de `frames_root` / `flow_root`.
- Construit la liste `samples = [(video_arcname, class_idx), …]` en parsant les
  clés de l'index frames (pas le filesystem).
- Charge un `TarReader` flow seulement si `mode in {"flow", "two_stream"}`.
- `_load_rgb` / `_load_gray` : `Image.open(io.BytesIO(reader.read(name)))`.

### CLI

```bash
uv run main.py --model CNN --archives archives
uv run debug_train.py     # ARCHIVES_ROOT="archives" en constante
```

`main.py` n'a plus `--frames-root` / `--flow-root`, juste `--archives`.

### Workflow de transfert

```bash
uv run pack_dataset.py                          # local : ~5-10 min
rsync -P archives/ user@autremachine:/path/     # 12 fichiers au lieu de 587k
uv run main.py --model CNN_rgb                  # idem sur les deux machines
```

## Nettoyage (`clean_dataset.py`)

Les sources avaient deux pathologies, désormais purgées :

1. **Pollution flow dans frames/** : `compute_optical_flow.py` a écrit certains
   `flow_*.jpg` directement dans `frames/test/<video>/`. 274 fichiers concernés,
   tous bit-exact identiques aux versions correctes côté `optical_flow/` →
   suppression sûre.
2. **Vidéos nativement incomplètes** : 11 vidéos sources cassées (pas leurs
   4 frames complètes) → suppression du dossier des deux côtés. Invariant
   après cleanup : **toute vidéo restante a ses 4 frames ET ses 6 flow**.

```bash
uv run clean_dataset.py            # dry-run : montre ce qui serait fait
uv run clean_dataset.py --apply    # applique
uv run pack_dataset.py             # re-pack ensuite
```

À relancer si tu re-télécharges du dataset ou regénères du flow.

## Modèles (`models_code/`)

Convention : un fichier = un modèle. Doit exposer :
- `INPUT_MODE` (str) : `"rgb_first"`, `"rgb_stack"`, `"flow"`, ou `"two_stream"`
- `build(num_classes, **kwargs) -> nn.Module`

`main.py --model X` charge `models_code/X.py` dynamiquement, lit son
`INPUT_MODE` pour configurer le `SmthSmthDataset` correspondant, puis appelle
`build(num_classes)`. Aucun registre central — pour ajouter un modèle, il
suffit de créer un nouveau fichier respectant la convention. `debug_train.py`
itère automatiquement sur tous les modèles de `models_code/`.

Combien d'images chaque modèle voit par vidéo :
- `CNN_rgb` : **1 frame** (la première) — apparence statique
- `CNN_flow` : **6 cartes flow** (toutes les paires x/y empilées en canaux)
- `CNN_two_stream` / `TSM_two_stream` / `R2plus1D_two_stream` : 4 frames + 6 flow

Pour une comparaison RGB-flow apples-to-apples côté quantité d'info, créer un
`CNN_rgb_stack.py` (mode `"rgb_stack"`, 12 canaux = 4 frames empilées) ou un
`CNN_flow_first.py` (1 paire de flow = 2 canaux).

## Entraînement et rapport

```bash
uv run main.py --model CNN_two_stream                # default : --save-dir runs
uv run main.py --model CNN_rgb --epochs 30 --lr 5e-4
```

Chaque run crée `runs/<model>_<YYYYMMDD-HHMMSS>/` (jamais écrasé, suffixe
`-2/-3` en cas de collision dans la même seconde). Y sont écrits :

- `config.json` au démarrage (tous les `args`, versions, GPU, git commit, nb
  params, taille dataset)
- `history.csv` au fil des epochs (append + flush — consultable pendant le run)
- `best.pth` à chaque amélioration de val_top1
- `summary.json` à la fin (best/final, total_time, per-class precision/recall/f1
  + confusion matrix calculés sur le best checkpoint)

À la fin, `main.py` appelle automatiquement `report_run.generate_report` qui
imprime un récap (hyperparams, top/flop classes par F1, etc.) et sauve
trois figures dans le dossier du run : `report_curves.png`,
`report_per_class.png`, `report_confusion.png`. Hook tolérant : si la
génération du rapport plante, le training reste sauvegardé.

Pour régénérer manuellement (ou sur un run hérité d'une autre machine) :

```bash
uv run report_run.py runs/CNN_two_stream_20260501-143022
uv run report_run.py runs/<dir> --frames-root /chemin/vers/frames
```

Les noms de classes pour les figures sont récupérés depuis
`frames/train/<NNN_NomClasse>/`. Si `frames/` n'est pas accessible, fallback en
labels numériques `class_NN`.

## Stack technique

D'après `requirements.txt` :

- **PyTorch 2.11** + **CUDA 13** + **cuDNN 9.19** → entraînement GPU prévu.
- **opencv-contrib-python 4.13** pour TVL1.
- **Optuna 4.8** → tuning d'hyperparamètres prévu.
- **Alembic + SQLAlchemy** → probablement pour persister les études Optuna.
- matplotlib, pandas, tqdm pour l'analyse / suivi.

## Pistes naturelles pour la suite

1. **Dataset PyTorch** qui charge, par vidéo : un tenseur RGB `(T=4, 3, H, W)`
   et un tenseur flux `(T=3, 2, H, W)` (flow_x + flow_y empilés).
2. **Modèle baseline** : two-stream simple
   - Branche RGB : ResNet 2D appliqué à chaque frame puis pooling temporel,
     ou un (2+1)D / 3D CNN léger.
   - Branche flux : même idée sur les 6 cartes (3 paires × 2 composantes).
   - Fusion tardive (concat + FC) → 32 logits.
3. **Mapping des labels** : construire `class_to_idx` en énumérant les 32
   dossiers triés (l'index 27 saute, donc `idx=27` correspondra au dossier
   `028_…`).
4. **Sortie test** : prédire pour chaque vidéo du split `test`
   (lu depuis `archives/frames_test.tar`) → CSV / JSON
   `video_id,class_id` (ou nom de classe).
5. **Optuna** : search sur lr, taille de batch, dropout, poids de fusion
   RGB/flow.

## Conventions / pièges à retenir

- Ne pas supposer 33 classes : il y en a **32** (027 manquant).
  Les modèles sortent actuellement **33 logits** (`--num-classes=33` par défaut
  dans `main.py`) — ce n'est pas un bug, l'utilisateur ajoutera plus tard le
  dossier de la classe 027 manquante. En attendant, l'index 27 n'apparaît
  jamais comme cible donc cette sortie reste simplement inutilisée.
- `test/` n'a pas de labels — toute évaluation pendant le dev se fait sur
  `val/`.
- Les 4 frames sont déjà sous-échantillonnées de la vidéo originale ; le
  modèle doit donc gérer une représentation temporelle très courte.
- Les flux JPG perdent un peu de précision (compression) — c'est volontaire
  et standard, mais ne pas re-clipper à la lecture.
- **Source de vérité au runtime = `archives/`**, pas `frames/` / `optical_flow/`.
  Les dossiers d'origine ne servent plus qu'à régénérer les tars via
  `pack_dataset.py`. Si tu modifies les frames/flows, repacke ensuite.
- Si `pack_dataset.py` est ré-exécuté, il **écrase** les tars existants — pas
  d'append incrémental.
