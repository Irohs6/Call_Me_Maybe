# call me maybe

## Introduction au function calling dans les LLM

**Résumé :** Les LLM parlent-ils le langage des ordinateurs ? Nous allons le découvrir.  
**Réalisé en collaboration avec :** @ldevelle, @pcamaren, @crfernan  
**Version :** 1.7

> Traduction française du sujet officiel. En cas de doute lors d'une évaluation, la version anglaise officielle fait foi.

## Sommaire

- Chapitre I — Avant-propos
- Chapitre II — Instructions concernant l’IA
- Chapitre III — Introduction
- Chapitre IV — Instructions communes
- Chapitre V — Partie obligatoire
- Chapitre VI — Exigences du README
- Chapitre VII — Partie bonus
- Chapitre VIII — Rendu et peer review

---

## Chapitre I — Avant-propos

Les ingénieurs romains gravaient des tablettes de pierre avec des grilles parfaitement organisées afin de suivre les livraisons de grain dans tout l’empire, pour qu’aucune livraison ne soit perdue à cause d’une mauvaise écriture. Au XIXe siècle, les marins consignaient les courants océaniques dans des tableaux structurés si précis que certains sont encore utilisés aujourd’hui pour la navigation.

Les premières prévisions météorologiques étaient envoyées par télégramme au moyen d’un code fixe : quelques nombres pouvaient décrire un ciel entier. Dans les années 1960, le centre de contrôle de mission de la NASA utilisait des organigrammes plastifiés qui expliquaient précisément la signification de chaque voyant, quelle que soit l’équipe en service. Les scanners de codes-barres des supermarchés traduisaient déjà des bandes noires et blanches en données d’inventaire bien avant que la plupart des foyers ne possèdent un ordinateur. Même les apiculteurs utilisent depuis des décennies des formulaires standardisés pour enregistrer la santé des ruches, le flux de nectar et la lignée des reines dans de petites cases faciles à lire d’un coup d’œil.

Les humains ont toujours créé des structures pour rendre l’information fiable, partageable et exploitable. Aujourd’hui, l’objectif est de faire parler à l’IA le langage des ordinateurs.

---

## Chapitre II — Instructions concernant l’IA

### Contexte

Au cours de votre apprentissage, l’IA peut vous aider pour de nombreuses tâches. Prenez le temps d’explorer les différentes capacités des outils d’IA et la manière dont ils peuvent soutenir votre travail. Cependant, utilisez-les toujours avec prudence et évaluez leurs résultats de manière critique. Qu’il s’agisse de code, de documentation, d’idées ou d’explications techniques, vous ne pouvez jamais être totalement certain que votre question était parfaitement formulée ou que le contenu généré est exact. Vos pairs sont une ressource importante pour éviter les erreurs et les angles morts.

### Message principal

- Utilisez l’IA pour réduire les tâches répétitives ou fastidieuses.
- Développez vos compétences de prompting, en programmation comme en dehors du code, car elles seront utiles pour votre future carrière.
- Apprenez comment fonctionnent les systèmes d’IA afin de mieux anticiper les risques, biais et problèmes éthiques courants.
- Continuez à développer vos compétences techniques et humaines en travaillant avec vos pairs.
- N’utilisez du contenu généré par IA que si vous le comprenez entièrement et pouvez en assumer la responsabilité.

### Règles pour les étudiants

- Prenez le temps d’explorer les outils d’IA et de comprendre leur fonctionnement afin de les utiliser de manière éthique et de limiter les biais.
- Réfléchissez à votre problème avant de rédiger un prompt : cela vous aidera à écrire des demandes plus claires, plus détaillées et plus pertinentes avec un vocabulaire précis.
- Prenez l’habitude de vérifier, relire, remettre en question et tester systématiquement tout ce qui est généré par une IA.
- Demandez toujours l’avis de vos pairs : ne reposez pas uniquement sur votre propre validation.

### Résultats attendus de la phase

- Développer des compétences de prompting générales et spécifiques à un domaine.
- Améliorer votre productivité grâce à une utilisation efficace des outils d’IA.
- Continuer à renforcer votre raisonnement informatique, votre résolution de problèmes, votre adaptabilité et votre collaboration.

### Commentaires et exemples

- Vous rencontrerez régulièrement des situations — examens, évaluations, etc. — où vous devrez démontrer une compréhension réelle. Soyez prêt et continuez à développer vos compétences techniques et relationnelles.
- Expliquer votre raisonnement et débattre avec vos pairs permet souvent de révéler des lacunes dans votre compréhension. Faites de l’apprentissage entre pairs une priorité.
- Les outils d’IA manquent souvent de votre contexte spécifique et ont tendance à produire des réponses génériques. Vos pairs, qui partagent votre environnement, peuvent apporter des points de vue plus pertinents et précis.
- Là où une IA tend à produire la réponse la plus probable, vos pairs peuvent proposer des perspectives différentes et des nuances utiles. Utilisez-les comme contrôle qualité.

**Bonne pratique :** demander à l’IA comment tester une fonction de tri, essayer les idées proposées, puis revoir les résultats avec un pair afin d’améliorer l’approche.

**Mauvaise pratique :** demander à l’IA d’écrire une fonction entière, la copier-coller, puis être incapable d’expliquer son fonctionnement pendant la peer-evaluation.

**Bonne pratique :** utiliser l’IA pour aider à concevoir un parser, puis revoir la logique avec un pair, détecter des bugs et réécrire ensemble la solution afin de la comprendre complètement.

**Mauvaise pratique :** laisser un assistant de code générer une partie essentielle du projet, constater que cela compile, mais être incapable d’expliquer le comportement pendant l’évaluation.

---

## Chapitre III — Introduction

### III.1 Qu’est-ce que le Function Calling ?

Les grands modèles de langage (LLM) sont très performants pour comprendre et générer du langage humain, mais ils ne produisent pas naturellement des sorties structurées directement exécutables par une machine. Le function calling comble cette lacune en transformant des requêtes en langage naturel en appels de fonctions précis avec des arguments typés.

Exemple :

Utilisateur :

```text
What is the sum of 40 and 2?
```

Réponse classique d’un LLM :

```text
The sum of 40 and 2 is 42.
```

Système de function calling :

```json
{
  "function": "add_numbers",
  "arguments": {
    "a": 40.0,
    "b": 2.0
  }
}
```

Le système de function calling ne répond donc pas directement à la question. Il fournit les éléments nécessaires pour résoudre le problème : le bon nom de fonction et les bons arguments avec les types appropriés.

### III.2 Pourquoi est-ce important ?

Le function calling permet aux LLM de :

- Interagir avec des systèmes externes : appeler des API, interroger des bases de données, contrôler des appareils.
- Exécuter du code : calculs, transformations de données, opérations sur des fichiers.
- Enchaîner des opérations : découper des tâches complexes en étapes exécutables.
- Produire des sorties structurées : JSON, XML ou autres formats lisibles par une machine.
- Extraire des données structurées à partir de texte non structuré. Par exemple, à partir d’un livre, extraire des champs comme le nom, le sexe et l’âge du protagoniste.

Cette technologie alimente les assistants IA modernes, les outils de génération de code et les agents autonomes. Elle permet aussi l’extraction automatique d’informations et la structuration de connaissances à partir de texte brut.

### III.3 Le défi

Les petits modèles de langage, comme le modèle de 0,6 milliard de paramètres utilisé dans ce projet, sont notoirement peu fiables lorsqu’il s’agit de produire des sorties structurées. Si on leur demande simplement de produire du JSON, ils peuvent n’y parvenir qu’environ 30 % du temps. Pourtant, des systèmes de production dépassent 99 % de fiabilité avec les mêmes petits modèles.

La solution repose sur le **constrained decoding**, une technique qui guide la sortie du modèle token par token afin de garantir une structure valide, sans se reposer uniquement sur le prompting.

---

## Chapitre IV — Instructions communes

### IV.1 Règles générales

- Le projet doit être écrit en **Python 3.10 ou supérieur**.
- Le projet doit respecter la norme **flake8**.
- Les fonctions doivent gérer correctement les exceptions afin d’éviter les crashs. Utilisez des blocs `try-except` pour les erreurs potentielles. Préférez les context managers pour les ressources comme les fichiers ou connexions. Un crash dû à une exception non gérée pendant l’évaluation rendra le projet non fonctionnel.
- Toutes les ressources, comme les fichiers et connexions réseau, doivent être correctement gérées afin d’éviter les fuites. Utilisez des context managers lorsque cela est possible.
- Le code doit inclure des annotations de type pour les paramètres, les valeurs de retour et les variables lorsque cela est pertinent. Utilisez **mypy** pour la vérification statique. Toutes les fonctions doivent passer mypy sans erreur.
- Les fonctions et classes doivent inclure des docstrings conformes à **PEP 257** (style Google ou NumPy, par exemple) décrivant leur objectif, leurs paramètres et leurs valeurs de retour.

### IV.2 Makefile

Le projet doit inclure un `Makefile` automatisant les tâches courantes. Il doit contenir les règles suivantes :

- `install` : installer les dépendances avec `pip`, `uv`, `pipx` ou un autre gestionnaire de paquets.
- `run` : exécuter le script principal du projet.
- `debug` : exécuter le script principal en mode debug avec le débogueur Python intégré, par exemple `pdb`.
- `clean` : supprimer les fichiers temporaires et caches comme `__pycache__` et `.mypy_cache`.
- `lint` : exécuter :

```bash
flake8 .
mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
  --disallow-untyped-defs --check-untyped-defs
```

- `lint-strict` *(optionnel)* : exécuter :

```bash
flake8 .
mypy . --strict
```

### IV.3 Consignes supplémentaires

- Créez des programmes de test pour vérifier le fonctionnement du projet. Ils ne sont pas rendus ni notés. Vous pouvez utiliser `pytest` ou `unittest`, en couvrant notamment les cas limites.
- Ajoutez un fichier `.gitignore` pour exclure les artefacts Python.
- L’utilisation d’un environnement virtuel est recommandée pour isoler les dépendances.

### IV.3.1 Exigences supplémentaires

- Toutes les classes doivent utiliser **Pydantic** pour la validation.
- Les paquets `numpy` et `json` sont autorisés.
- L’utilisation de `dspy` ou de tout paquet similaire est strictement interdite, notamment `pytorch`, `huggingface`, `transformers`, `outlines`, etc.
- Le modèle à utiliser est :
  - `Qwen/Qwen3-0.6B` par défaut.
  - D’autres modèles peuvent être utilisés tant que le projet fonctionne avec `Qwen/Qwen3-0.6B`.
- La fonction à appeler doit être choisie par le **LLM**, et non à l’aide d’heuristiques ou de toute autre « magie médiévale ».
- Il est interdit d’utiliser des méthodes ou attributs privés du package `llm_sdk`.
- Vous devez créer un environnement virtuel et installer `numpy` et `pydantic` avec `uv`. Le package `llm_sdk` peut être copié dans le même répertoire parent que `src`.
- Le reviewer et la moulinette exécuteront simplement `uv sync`.
- Toutes les erreurs doivent être gérées proprement. Le programme ne doit jamais planter de manière inattendue et doit toujours fournir des messages d’erreur clairs.

### IV.3.2 Utilisation

Le programme doit être lancé avec :

```bash
uv run python -m src [--functions_definition <function_definition_file>] \
  [--input <input_file>] [--output <output_file>]
```

Par défaut, le programme lit les fichiers d’entrée dans `data/input/` et écrit les fichiers de sortie dans `data/output/`.

Des chemins personnalisés peuvent être fournis avec `--input`, `--output` et `--functions_definition`.

Exemple :

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

---

## Chapitre V — Partie obligatoire

### V.1 Résumé

Dans ce projet, vous devez créer un outil de function calling qui transforme des prompts en langage naturel en appels de fonctions structurés.

Pour une question telle que :

```text
What is the sum of 40 and 2?
```

la solution ne doit pas répondre `42`. Elle doit produire :

- Le nom de la fonction : `fn_add_numbers`
- Les arguments : `{"a": 40.0, "b": 2.0}`

L’implémentation doit utiliser le **constrained decoding** afin de garantir **100 % de JSON valide**, tout en fournissant une fiabilité proche de la perfection avec un petit modèle de 0,6 milliard de paramètres.

### V.2 Fichiers d’entrée

La solution doit traiter deux fichiers situés dans `data/input/`.

#### `function_calling_tests.json`

Il contient un tableau JSON de prompts en langage naturel.

Exemple :

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "What is the sum of 265 and 345?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Greet john"},
  {"prompt": "Reverse the string 'hello'"}
]
```

#### `functions_definition.json`

Il contient les fonctions disponibles. Chaque fonction comprend :

- son nom ;
- les noms et types de ses arguments ;
- son type de retour ;
- sa description.

Exemple :

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {"type": "string"}
    },
    "returns": {"type": "string"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

Ces exemples indiquent le niveau de complexité attendu. Cependant, votre solution sera testée avec d’autres prompts et d’autres ensembles de fonctions.

Vous devez gérer correctement les erreurs JSON des fichiers d’entrée, car ils peuvent être absents ou contenir du JSON invalide.

### V.3 Interaction avec le LLM

#### V.3.1 Le SDK du LLM

Le projet fournit une classe wrapper `Small_LLM_Model` dans le package `llm_sdk` pour interagir avec le modèle.

Le SDK fournit plusieurs méthodes essentielles :

- `get_logits_from_input_ids(input_ids: List[int]) -> List[float]` : prend une liste d’identifiants de tokens et renvoie les logits produits par le modèle.
- `get_path_to_vocab_file() -> str` : renvoie le chemin du fichier de vocabulaire associant les identifiants aux tokens.
- `encode(text: str) -> Tensor` : encode une chaîne de caractères en identifiants de tokens avec le tokenizer du modèle.
- `decode(token_ids: List[int]) -> str` *(optionnel)* : décode une liste d’identifiants de tokens en texte.

#### V.3.2 Pipeline de génération

Le processus de génération suit les étapes suivantes :

1. **Prompt** : la question en langage naturel.
2. **Tokenisation** : le texte est découpé en unités de sous-mots appelées tokens. Les tokenizers peuvent conserver les espaces, gérer la ponctuation et diviser les mots selon des algorithmes comme BPE ou SentencePiece.
3. **Input IDs** : les tokens sont convertis en identifiants numériques compris par le modèle.
4. **Traitement par le LLM** : le modèle traite ces nombres à travers son réseau neuronal.
5. **Logits** : le modèle produit des scores pour chaque prochain token possible.
6. **Sélection du token** : le prochain token est choisi à partir de ces scores, généralement celui ayant le score le plus élevé.

C’est à cette étape que le **constrained decoding** peut limiter les tokens possibles afin de garantir une structure déterminée, par exemple un JSON toujours valide.

Ce processus se répète token par token. Chaque token généré est ajouté au prompt, puis les étapes précédentes sont répétées jusqu’à l’obtention de la réponse complète.

Vue simplifiée :

```text
Prompt -> Tokenisation -> Input IDs -> LLM -> Logits -> Sélection du prochain token
```

#### V.3.3 Comprendre le Constrained Decoding

Les modèles de langage génèrent le texte un token à la fois. À chaque étape, le modèle produit une distribution de probabilités, représentée par des logits, sur tous les tokens possibles.

Le constrained decoding intervient avant la sélection du prochain token :

1. Le modèle produit les logits de tous les tokens possibles.
2. Vous déterminez quels tokens permettent de conserver à la fois un JSON valide et la conformité au schéma attendu.
3. Vous donnez une valeur de moins l’infini aux logits des tokens invalides, c’est-à-dire ceux qui casseraient la structure ou le schéma.
4. Vous ne sélectionnez ensuite qu’un token parmi ceux qui restent autorisés.

Dans ce projet, le constrained decoding doit garantir non seulement un JSON syntaxiquement valide, mais aussi le respect du schéma. Par exemple, si un champ est défini comme un nombre dans `functions_definition.json`, le décodeur doit empêcher la génération d’une valeur incompatible avec un nombre flottant.

Chaque token généré doit donc maintenir la validité structurelle et sémantique de la sortie. Le JSON produit doit toujours pouvoir être récupéré et parsé sans erreur.

> La solution ne doit **PAS** reposer sur l’espoir que le modèle produise spontanément du JSON correct. Mettre les définitions de fonctions dans le prompt et espérer une sortie structurée n’est pas considéré comme fiable et ne correspond pas à la compétence demandée dans ce projet.

Le fichier de vocabulaire JSON doit être utilisé pour faire la correspondance entre les tokens et leurs représentations textuelles. Cette correspondance est essentielle pour déterminer quels tokens restent valides à chaque étape de génération.

### V.4 Format du fichier de sortie

Le programme doit produire un unique fichier JSON :

```text
data/output/function_calls.json
```

Pour chaque prompt, ajoutez un objet JSON au tableau. Chaque objet doit contenir exactement les clés suivantes :

- `prompt` (`string`) : requête originale en langage naturel ;
- `name` (`string`) : nom de la fonction à appeler ;
- `parameters` (`object`) : tous les arguments requis avec les types corrects.

#### V.4.1 Exemple de sortie

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

#### V.4.2 Règles de validation

- Le fichier doit être un JSON valide : pas de virgule finale et pas de commentaire.
- Les clés et les types doivent correspondre exactement au schéma défini dans `functions_definition.json`.
- Aucune clé supplémentaire ni aucun texte explicatif ne doit apparaître dans la sortie.
- Tous les arguments requis doivent être présents.
- Le type de chaque argument doit correspondre à la définition de la fonction (`number`, `string`, `boolean`, etc.).

> Les fichiers d’entrée fournis peuvent changer pendant la peer review. Ne codez pas les réponses en dur à partir des exemples fournis.

### V.5 Performances et fiabilité

L’implémentation doit viser :

- **Précision quasi parfaite :** au moins 90 % de bonnes sélections de fonction et extractions d’arguments.
- **100 % de JSON valide :** chaque sortie doit être parsable et conforme au schéma.
- **Vitesse raisonnable :** traiter tous les prompts de test en moins de 5 minutes sur du matériel standard.
- **Gestion robuste des erreurs :** gérer proprement les entrées invalides, fichiers manquants et cas limites.

Le modèle `Qwen3-0.6B` ne possède qu’environ 500 millions de paramètres, mais un constrained decoding correctement conçu peut lui donner une fiabilité comparable à celle de modèles beaucoup plus grands. Cela démontre l’importance du guidage structurel par rapport à la seule taille du modèle.

### V.6 Tester votre implémentation

Pour vérifier que la solution fonctionne correctement :

1. Placez les fichiers d’entrée dans `data/input/`.
2. Lancez :

```bash
uv run python -m src [--functions_definition <function_definition_file>] \
  [--input <input_file>] [--output <output_file>]
```

3. Vérifiez que `output/function_calls.json` est créé.
4. Validez la structure et le contenu du JSON.
5. Vérifiez que les noms de fonctions et types d’arguments correspondent aux définitions.

Testez notamment :

- les chaînes vides ;
- les grands nombres ;
- les caractères spéciaux ;
- les mauvais types ;
- les prompts ambigus ;
- les fonctions ayant plusieurs paramètres.

---

## Chapitre VI — Exigences du README

Un fichier `README.md` doit être présent à la racine du dépôt Git. Son but est de permettre à toute personne ne connaissant pas le projet — pairs, staff, recruteurs, etc. — de comprendre rapidement le projet, comment l’exécuter et où trouver davantage d’informations.

Le README doit au minimum contenir :

- En toute première ligne, en italique :

```text
This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].
```

- Une section **Description** présentant clairement l’objectif et une vue d’ensemble du projet.
- Une section **Instructions** contenant les informations utiles pour la compilation, l’installation et/ou l’exécution.
- Une section **Resources** listant les références classiques liées au sujet — documentation, articles, tutoriels, etc. — ainsi qu’une description de l’utilisation de l’IA : pour quelles tâches et quelles parties du projet.

Des sections supplémentaires peuvent être requises selon le projet, par exemple des exemples d’utilisation, une liste de fonctionnalités ou des choix techniques.

Pour ce projet, le README doit également contenir :

- **Explication de l’algorithme :** détailler votre approche du constrained decoding.
- **Choix de conception :** expliquer les décisions principales prises dans l’implémentation.
- **Analyse des performances :** discuter de la précision, de la vitesse et de la fiabilité.
- **Difficultés rencontrées :** documenter les problèmes rencontrés et la manière dont ils ont été résolus.
- **Stratégie de test :** expliquer comment l’implémentation a été validée.
- **Exemples d’utilisation :** fournir des exemples clairs d’exécution du programme.

> Le `README.md` doit être écrit en anglais.

---

## Chapitre VII — Partie bonus

Les fonctionnalités bonus sont optionnelles et ne sont pas nécessaires pour valider le projet :

- Support de plusieurs modèles LLM au-delà de `Qwen/Qwen3-0.6B`.
- Réimplémentation du tokenizer afin d’éviter l’utilisation directe de `encode` et `decode` dans le code principal, en utilisant plutôt `get_logits_from_input_ids` et `get_path_to_vocab_file`.
- Mécanismes avancés de récupération après erreur.
- Optimisations de performance, par exemple cache et batching.
- Suite de tests complète.
- Visualisation du processus de génération.
- Support d’arguments de fonctions complexes et imbriqués.
- Implémentation publique des méthodes `encode` et éventuellement `decode` du tokenizer.
- Démonstration de l’intégration de l’encodage et du décodage avec le constrained decoding.

Les bonus doivent être réellement implémentés et fonctionnels, pas seulement décrits dans le README. Vous pouvez avoir à les démontrer pendant l’évaluation.

Vous pouvez utiliser d’autres fonctions et bibliothèques pour la partie bonus, tant que leur utilisation est justifiée pendant l’évaluation.

> La partie bonus ne sera évaluée que si la partie obligatoire est parfaite. « Parfaite » signifie qu’elle est entièrement terminée et fonctionne sans erreur. Si toutes les exigences obligatoires ne sont pas validées, les bonus ne seront pas évalués.

---

## Chapitre VIII — Rendu et peer review

Rendez le projet dans votre dépôt Git comme d’habitude. Seul le travail présent dans le dépôt sera évalué pendant la soutenance. Vérifiez bien le nom de vos fichiers.

Le dépôt doit contenir :

- le dossier `src/` avec l’implémentation ;
- `pyproject.toml` et `uv.lock` pour la gestion des dépendances ;
- le dossier `llm_sdk/`, copié depuis le package fourni ;
- le dossier `data/input/` avec des fichiers de test pour la démonstration ;
- un `README.md` complet ;
- tous les autres fichiers nécessaires à l’exécution.

> N’incluez pas le dossier `output/` dans le dépôt. Il sera généré pendant la peer review.

Pendant l’évaluation, une petite modification du projet peut parfois être demandée. Cela peut être un changement mineur de comportement, quelques lignes de code à écrire ou réécrire, ou une fonctionnalité simple à ajouter.

Cette étape peut ne pas s’appliquer à tous les projets, mais vous devez y être préparé si elle apparaît dans les consignes d’évaluation.

Le but est de vérifier votre compréhension réelle d’une partie du projet. La modification peut être réalisée dans l’environnement de développement de votre choix et devrait pouvoir être effectuée en quelques minutes, sauf si un délai spécifique est défini.

Il peut par exemple être demandé de modifier légèrement une fonction ou un script, changer un affichage ou adapter une structure de données pour stocker une nouvelle information.

Les détails exacts — portée, cible, etc. — seront indiqués dans les consignes d’évaluation et peuvent varier d’une évaluation à l’autre pour le même projet.
