# Rapport détaillé : function calling de Qwen

Mise à jour : **25 octobre 2026**. Modèle testé : **Qwen3-0.6B**.

Ce document rassemble les comparaisons, le fonctionnement et les limites de la version conservée dans le projet. Les résultats complets, les définitions et les réponses attendues sont archivés dans [resultats_function_calling.json](resultats_function_calling.json). Leur consultation ne dépend pas du dossier local d’essai.

## 1. Version actuellement conservée

Le programme utilise :

1. Une sélection A/B de la fonction.
2. La génération des arguments, dans un objet commun pour les fonctions ayant plusieurs paramètres tous textuels.
3. Une révision de certains motifs regex par le LLM, en conservant le texte source et le remplacement.

| Jeu | Résultat final |
|---|---:|
| Bonne fonction : 28 demandes claires, 14 fonctions | **27/28** |
| Appel complet : mêmes 28 demandes | **27/28** |
| Appels complets : 19 contrôles, les 6 fonctions fournies | **19/19** |
| Quatre regex de la campagne | **4/4** |
| Trois regex du fichier de requêtes fourni | **3/3** |
| Douze formulations regex supplémentaires | **9/12** |
| Huit autres formulations de contrôle | **6/8** |

Ces jeux se recoupent : les quatre regex font partie des 28 demandes ; les 19 contrôles réutilisent une partie des mêmes demandes. Les fractions ne doivent pas être additionnées comme des observations indépendantes.

Ces résultats décrivent les jeux archivés. Ils ne constituent pas une garantie sur toutes les formulations.

## 2. Fonctionnement de la sélection A/B

Le LLM reçoit la demande et deux fiches de fonctions, temporairement identifiées par A et B. Les descriptions, paramètres et vrais noms restent visibles.

Le code contraint la réponse aux deux étiquettes. Les scores du modèle déterminent la gagnante. Celle-ci est comparée à la fonction suivante jusqu’à la fin du catalogue.

~~~text
F1 contre F2 → gagnante
gagnante contre F3 → nouvelle gagnante
nouvelle gagnante contre F4 → ...
~~~

Avec N fonctions, il faut N − 1 comparaisons : 13 pour 14 fonctions. Ce n’est pas un tri complet. Une comparaison peut demander plusieurs requêtes de logits selon la tokenisation des étiquettes.

Chaque comparaison dispose d’un nouveau contexte contenant la demande et les deux fonctions du tour. Le LLM ne conserve pas l’historique des tours ; le programme conserve seulement la gagnante.

**La fonction est choisie par le LLM.** Aucune règle du programme n’associe un mot de la demande à une fonction.

Le jugement du modèle peut être incohérent entre les tours : l’ordre du catalogue peut influencer la gagnante.

## 3. Comparaison avec le template officiel et tous les outils ensemble

La variante officielle testée utilise le chat template de Qwen3-0.6B en mode non-thinking. Le contexte a été comparé à celui produit par le tokenizer officiel, y compris les IDs de tokens.

Le décodeur du projet ajoute un préfixe d’appel et contraint le nom au catalogue. Il conserve une fonction unique par demande et le générateur d’arguments de cette campagne. Ce test ne mesure pas les appels multiples ou l’abstention libres.

**Les descriptions et le générateur d’arguments étaient identiques entre les méthodes de sélection.** Les essais distincts avec descriptions réécrites sont exclus du bilan.

| Jeu | Bonne fonction : officiel | A/B | Appel complet : officiel | A/B |
|---|---:|---:|---:|---:|
| 28 demandes claires, 14 fonctions | 23/28 | **27/28** | 19/28 | **23/28** |
| 19 contrôles, 6 fonctions fournies | 19/19 | 19/19 | 15/19 | 15/19 |
| Mêmes 19 demandes, 14 fonctions | 15/19 | **19/19** | 11/19 | **15/19** |

Les scores d’appels complets correspondent au générateur d’arguments utilisé à l’époque. L’écart avec les 27/28 de la version finale ne provient donc pas seulement de la sélection A/B.

Sur cinq demandes d’inversion du premier jeu, la sélection officielle réussit 2/5 et A/B 5/5. Sur six autres formulations, testées dans les deux ordres du catalogue, les résultats sont 8/12 et 11/12.

A/B a été conservée sur cette base. Le tournoi est une adaptation expérimentale du projet ; il ne s’agit pas d’une recommandation du guide officiel.

## 4. Génération commune des arguments

L’ancienne méthode générait les arguments textuels séparément, en reconstruisant le contexte avec les valeurs précédentes. Elle pouvait confondre motif et remplacement, copier un exemple de nombre comme regex ou ajouter des espaces au remplacement.

Les fonctions ayant plusieurs paramètres tous de type string complètent désormais un même objet d’arguments, sous contrainte JSON. La grammaire impose :

- les clés attendues, une seule fois chacune ;
- toutes les clés obligatoires ;
- des valeurs de type chaîne ;
- des échappements JSON valides.

Les fonctions à un seul argument texte et la méthode numérique conservent leur génération précédente.

La grammaire sécurise la structure. Elle ne prouve pas qu’une chaîne contient un motif pertinent.

| Jeu, sélection A/B identique | Arguments séparés | Arguments communs |
|---|---:|---:|
| 28 demandes claires : appel complet | 23/28 | **25/28** |
| 19 contrôles : appel complet | 15/19 | **17/19** |
| Quatre regex | 0/4 | **2/4** |
| Trois regex du fichier fourni | 2/3 | **3/3** |
| Douze formulations supplémentaires | 2/12 | **8/12** |

## 5. Révision des motifs regex

La génération commune laissait notamment une expression trop large pour les voyelles et un point interprété comme joker.

Pour la signature source_string / regex / replacement, une révision examine les motifs contenant un joker ou une ancre non échappés hors des classes, ainsi que ceux pouvant correspondre au texte vide.

À partir du motif généré, elle propose :

1. Le motif original.
2. Son interprétation littérale, encodée par re.escape.
3. Les classes de caractères présentes dans le motif, prises individuellement.

Le LLM choisit entre ces interprétations par comparaisons A/B, avec la demande originale. Le code ne déduit pas l’intention au moyen de mots-clés.

Les identifiants internes de ces choix ne remplacent pas la fonction finale. Le programme n’exécute aucune fonction de substitution. Le texte source et le remplacement sont conservés.

Il n’existe pas de table de motifs prédéfinis pour les tests : les caractères et les classes viennent de la sortie du modèle. Les variantes exploratoires de planification et de code Python, qui provoquaient des régressions, ont été écartées.

| Jeu | Avant la révision | Après |
|---|---:|---:|
| 28 demandes claires : appel complet | 25/28 | **27/28** |
| 19 contrôles : appel complet | 17/19 | **19/19** |
| Quatre regex de la campagne | 2/4 | **4/4** |
| Trois regex du fichier fourni | 3/3 | **3/3** |
| Douze formulations supplémentaires | 8/12 | **9/12** |
| Huit nouvelles demandes | 4/8 | **6/8** |

Sur les voyelles de banana, le modèle retient désormais la classe [aeiou]. Sur le point littéral, il retient un point échappé. Les jokers et ancres volontairement demandés ont été inclus dans les vérifications supplémentaires.

## 6. Validation et intégrité des données

Les variantes comparées utilisent les mêmes requêtes, réponses attendues, schémas et ordres. Les fichiers fournis et les fixtures figées sont vérifiés par SHA256 ; les deux fichiers d’entrée sont aussi comparés octet par octet.

Les 19 contrôles utilisent exactement les six définitions fournies, y compris la description de multiplication disant « deux nombres » alors que le schéma impose quatre paramètres.

Le catalogue de 14 fonctions provient de la campagne précédente. Il ajoute huit fonctions et avait déjà une description de multiplication à quatre opérandes. Il reste identique entre les variantes. Il ne faut pas le confondre avec le fichier fourni.

Les critères sont :

- bonne sélection : nom attendu ;
- appel complet : bonne fonction et bons arguments ;
- permutations acceptées pour addition et multiplication ;
- ordre conservé pour soustraction et division ;
- casse et espaces des textes vérifiés ;
- regex évaluées par le résultat de re.sub dans le banc d’essai, avec texte source exact : des motifs équivalents sont acceptés.

Le programme principal ne réalise pas cette substitution. Elle sert seulement à la validation.

Les huit demandes ambiguës, la demande incomplète et la demande hors catalogue sont conservées comme observations, sans score binaire de réussite.

La dernière comparaison comprend 126 exécutions principales et de contrôle, 24 exécutions des entrées fournies, 24 vérifications des douze formulations supplémentaires et 16 vérifications nouvelles : **190 exécutions avant/après**. Dans les contrôles complémentaires, la fonction de substitution est fixée pour isoler les arguments.

## 7. Limites observées

### Sélection

Le partage de 21 biscuits entre 3 personnes choisit encore une addition au lieu de la division. C’est le seul échec des 28 demandes claires dans la version finale.

A/B reste sensible à l’ordre et nécessite davantage de comparaisons que la sélection globale.

### Regex

Les cas d’origine sur les voyelles et le point sont corrigés. D’autres formulations échouent encore :

- remplacement de tous les nombres réduit au motif littéral 607 ;
- astérisque littéral interprété comme une expression trop large ;
- absence de frontières lors du remplacement d’un mot entier ;
- texte lowercase vowels produit comme motif ;
- classe négative au lieu de positive pour certaines voyelles.

La révision ne choisit que parmi les interprétations qu’elle construit. Elle n’est pas un validateur sémantique universel.

### Nombres et demandes non réalisables

Les quatre notations numériques difficiles restent incorrectes : nombres en lettres, notation scientifique, décimales sans zéro initial et séparateurs de milliers.

Le programme force une fonction unique. Il n’a pas encore de voie de clarification ou d’abstention pour les demandes ambiguës, incomplètes ou hors catalogue.

La boucle actuelle ne génère pas explicitement les arguments booléens ; ils ne sont pas couverts par les scores ci-dessus.

### Portée des mesures

Les essais utilisent le modèle local dans le venv du projet, sur le GPU de cette machine. Ils n’ont pas été confirmés sur les Mac de 42. Le backend et la précision du SDK peuvent modifier les scores de tokens et certaines sorties.

## 8. Temps mesurés

| Comparaison, 28 cas clairs hors chargement | Avant | Après |
|---|---:|---:|
| Sélection officielle → A/B, arguments identiques | 19,32 s | 36,64 s |
| Arguments séparés → communs | 35,89 s | 41,35 s |
| Ajout de la révision regex | 40,65 s | 40,95 s |

Ces campagnes sont distinctes, sans répétitions statistiques. Elles ne permettent pas de déduire un temps sur CPU ou sur Mac. La petite différence de la dernière ligne ne prouve pas un coût supplémentaire négligeable dans tous les cas.

## 9. Nettoyage du code et vérifications

Nettoyage réalisé sans modifier les décisions ni les sorties :

- retrait de méthodes inutilisées : ancien encodage des noms, extraction de chaînes candidates et ancien décodeur de chaînes candidates ;
- retrait de l’ancien contexte descriptif inutilisé ;
- schéma de fonction centralisé dans ContextBuilder.tool_definition ;
- annotations des candidats numériques corrigées : clés entières, couples texte/tokens ;
- paramètre inutilisé de generate_number supprimé ;
- imports regroupés, docstrings ajoutées, constantes et variable de boucle inutilisées retirées de l’entrée CLI ;
- exception redondante simplifiée : JSONDecodeError est déjà couvert par ValueError.

Les contextes envoyés au LLM sont identiques à ceux d’avant nettoyage. La configuration des dépendances, le SDK et les fichiers d’entrée sont conservés. Aucune modification CUDA n’est ajoutée aux sources portables.

Vérifications :

- **1 845 comparaisons de contextes strictement identiques** ;
- **16 tests unitaires** : révision regex, grammaire et décodage JSON ;
- **95 sorties de modèle identiques** à l’état avant nettoyage, y compris les erreurs observées ;
- compilation syntaxique et git diff --check.

Le nettoyage ne constitue pas une nouvelle amélioration de précision. Mypy n’a pas été exécuté dans cette validation.

## 10. Fichiers et consultation

| Fichier | Rôle |
|---|---|
| [function_selector.py](../src/function_selector.py) | Sélection A/B et génération des arguments |
| [context_builder.py](../src/model/context_builder.py) | Schémas et contextes |
| [string_arguments.py](../src/model/string_arguments.py) | Grammaire incrémentale |
| [regex_review.py](../src/model/regex_review.py) | Interprétations des motifs |
| [resultats_function_calling.json](resultats_function_calling.json) | Résultats, fixtures et validation |
| [README.md](README.md) | Index de la documentation |

Le rapport et ses données sont directement dans docs/. Supprimer les scripts, caches et environnements locaux d’essai ne les supprime pas.

Depuis la racine du projet :

~~~bash
uv run python -m src
~~~

Les options existantes permettent de choisir les entrées et la sortie :

~~~bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
~~~

Sur cette machine, le lancement GPU local reste :

~~~bash
bash .local-cuda/run.sh
~~~

Ce script et son environnement restent locaux. Les scores du rapport sont ceux des jeux archivés, pas un résultat promis pour toute entrée.

## 11. Références utilisées

- [Documentation Qwen : Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Modèle Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Chat template de Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)

Le guide recommande Hermes pour Qwen3 et décrit les fonctions avec JSON Schema. Notre tournoi A/B et la révision des motifs sont des adaptations du projet. Un template correct ne garantit pas la justesse des arguments.

## Annexe A — Les 28 demandes claires

| Cas | Demande | Fonction produite | Arguments | Appel correct |
|---|---|---|---|---|
| add-story | I had 14 marbles and received 7 more. How many do I have in total? | fn_add_numbers | `{"a":14.0,"b":7.0}` | oui |
| add-decimals | Calculate -3.5 plus 2.25. | fn_add_numbers | `{"a":2.25,"b":-3.5}` | oui |
| add-distractor | Invoice reference 2026. Add 17 and 8; ignore the reference number. | fn_add_numbers | `{"a":17.0,"b":8.0}` | oui |
| multiply-four | Multiply 2, 3, 4 and 5 together. | fn_multiply_numbers | `{"a":2.0,"b":3.0,"c":4.0,"d":5.0}` | oui |
| multiply-zero | Find the product of -2, 0.5, 0 and 4. | fn_multiply_numbers | `{"a":0.5,"b":0.0,"c":4.0,"d":-2.0}` | oui |
| multiply-repeated | Multiply 2 by 2 by 3 by 3. | fn_multiply_numbers | `{"a":2.0,"b":2.0,"c":3.0,"d":3.0}` | oui |
| sqrt-paraphrase | Which positive number, multiplied by itself, gives 81? | fn_get_square_root | `{"a":81.0}` | oui |
| sqrt-french | Calcule la racine carrée de 225. | fn_get_square_root | `{"a":225.0}` | oui |
| greet-paraphrase | Say hello to Alice. | fn_greet | `{"name":"Alice"}` | oui |
| greet-distractor | Greet the person named 'Square Root'. | fn_greet | `{"name":"Square Root"}` | oui |
| reverse-paraphrase | Write 'Desserts' backwards. | fn_reverse_string | `{"s":"Desserts"}` | oui |
| reverse-distractor | Reverse the text 'multiply 2 by 3'; do not calculate it. | fn_reverse_string | `{"s":"multiply 2 by 3"}` | oui |
| reverse-unicode | Reverse the string 'été 🐱'. | fn_reverse_string | `{"s":"été 🐱"}` | oui |
| reverse-spaces | Reverse the string '  a b  ', keeping its spaces as part of the input. | fn_reverse_string | `{"s":"  a b  "}` | oui |
| subtract-order | Subtract 5 from 20. | fn_subtract_numbers | `{"a":20.0,"b":5.0}` | oui |
| divide-story | Share 21 cookies equally among 3 people. How many cookies per person? | fn_add_numbers | `{"a":21.0,"b":3.0}` | non |
| uppercase | Make 'Hello, world!' ALL CAPS. | fn_uppercase_text | `{"text":"Hello, world!"}` | oui |
| lowercase | Convert 'MiXeD CaSe' to lowercase. | fn_lowercase_text | `{"text":"MiXeD CaSe"}` | oui |
| count-words | How many whitespace-separated words are in 'one two three'? | fn_count_words | `{"text":"one two three"}` | oui |
| even | Is 14 even? | fn_is_even | `{"value":14.0}` | oui |
| temperature | Convert -10 degrees Celsius into Fahrenheit. | fn_celsius_to_fahrenheit | `{"celsius":-10.0}` | oui |
| reverse-words | Reverse the order of the words in 'red green blue', without reversing any word's letters. | fn_reverse_words | `{"text":"red green blue"}` | oui |
| regex-numbers | Replace every digit sequence in 'ID 12, batch 345' with '#'. | fn_substitute_string_with_regex | `{"source_string":"ID 12, batch 345","regex":"\\d+","replacement":"#"}` | oui |
| regex-vowels | Replace all lowercase vowels in 'banana' with '*'. | fn_substitute_string_with_regex | `{"source_string":"banana","regex":"[aeiou]","replacement":"*"}` | oui |
| regex-literal-dot | In 'a.b.a.b', replace each literal dot with '-'. | fn_substitute_string_with_regex | `{"source_string":"a.b.a.b","regex":"\\.","replacement":"-"}` | oui |
| subtract-french | Fais la différence entre 20 et 5, dans cet ordre. | fn_subtract_numbers | `{"a":20.0,"b":5.0}` | oui |
| regex-delete-spaces | Delete every whitespace character from ' a b  c '. | fn_substitute_string_with_regex | `{"source_string":" a b  c ","regex":"\\s+","replacement":""}` | oui |
| greet-case | Say hello to 'jean-luc'. | fn_greet | `{"name":"jean-luc"}` | oui |

Les fonctions et valeurs attendues sont conservées dans le JSON associé.

## Annexe B — Nombres difficiles et demandes non notées

### ambiguous-operation

Do something with 8 and 2.

**Nature :** ambiguous. No operation specified; no unique correct call.

~~~json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 8.0,
    "b": 2.0
  }
}
~~~

### ambiguous-reverse

Make 'hello world' backwards.

**Nature :** ambiguous. Characters or word order are both plausible.

~~~json
{
  "name": "fn_reverse_string",
  "parameters": {
    "s": "hello world"
  }
}
~~~

### ambiguous-case

Fix the case of 'hELLo'.

**Nature :** ambiguous. Target capitalization unspecified; title case is not available.

~~~json
{
  "name": "fn_uppercase_text",
  "parameters": {
    "text": "hELLo"
  }
}
~~~

### ambiguous-greeting

Greet John and Mary.

**Nature :** ambiguous. Two recipients, but only one scalar name per call.

~~~json
{
  "name": "fn_greet",
  "parameters": {
    "name": "John"
  }
}
~~~

### ambiguous-replacement

Replace 'cat' in 'cat dog cat'.

**Nature :** ambiguous. Replacement value is missing; should request clarification.

~~~json
{
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "cat dog cat",
    "regex": "cat",
    "replacement": ""
  }
}
~~~

### ambiguous-temperature

Convert 20 degrees.

**Nature :** ambiguous. Source and destination units unspecified.

~~~json
{
  "name": "fn_celsius_to_fahrenheit",
  "parameters": {
    "celsius": 20.0
  }
}
~~~

### ambiguous-chain

Add 2 and 3, then take the square root of the result.

**Nature :** ambiguous. Requires two operations; a single call cannot fulfill the request.

~~~json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
~~~

### ambiguous-two-actions

Reverse 'hello' and greet Alice.

**Nature :** ambiguous. Two requested actions; a single call cannot fulfill both.

~~~json
{
  "name": "fn_reverse_words",
  "parameters": {
    "text": "hello Alice"
  }
}
~~~

### edge-number-words

What is two plus three?

**Nature :** edge. Numbers written in words.

~~~json
{
  "name": "fn_add_numbers",
  "error": "ValueError: No number found in prompt",
  "error_stage": "arguments"
}
~~~

### edge-scientific

Add 1e3 and 2.

**Nature :** edge. Scientific notation.

~~~json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 1.0,
    "b": 2.0
  }
}
~~~

### edge-leading-decimal

Add .5 and 2.

**Nature :** edge. Decimal with no leading zero.

~~~json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 5.0,
    "b": 2.0
  }
}
~~~

### edge-thousands

Add 1,000 and 25.

**Nature :** edge. Thousands separator.

~~~json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 1.0,
    "b": 25.0
  }
}
~~~

### missing-multiply-inputs

Multiply 2 and 3.

**Nature :** incomplete. Four required arguments, only two operands; do not invent the others.

~~~json
{
  "name": "fn_multiply_numbers",
  "error": "ValueError: No number found in prompt",
  "error_stage": "arguments"
}
~~~

### unsupported-weather

What is the weather in Paris?

**Nature :** unsupported. No suitable function; should abstain or explain the limitation.

~~~json
{
  "name": "fn_greet",
  "parameters": {
    "name": "Paris"
  }
}
~~~


## Annexe C — Les quatre cas regex de la campagne

### regex-numbers

Replace every digit sequence in 'ID 12, batch 345' with '#'.

~~~json
{
  "source_string": "ID 12, batch 345",
  "regex": "\\d+",
  "replacement": "#"
}
~~~

**Résultat vérifié :** "ID #, batch #"

### regex-vowels

Replace all lowercase vowels in 'banana' with '*'.

~~~json
{
  "source_string": "banana",
  "regex": "[aeiou]",
  "replacement": "*"
}
~~~

**Résultat vérifié :** "b*n*n*"

### regex-literal-dot

In 'a.b.a.b', replace each literal dot with '-'.

~~~json
{
  "source_string": "a.b.a.b",
  "regex": "\\.",
  "replacement": "-"
}
~~~

**Résultat vérifié :** "a-b-a-b"

### regex-delete-spaces

Delete every whitespace character from ' a b  c '.

~~~json
{
  "source_string": " a b  c ",
  "regex": "\\s+",
  "replacement": ""
}
~~~

**Résultat vérifié :** "abc"


## Annexe D — Formulations supplémentaires

| Cas | Demande | Motif produit | Résultat vérifié | Correct |
|---|---|---|---|---|
| holdout-digits | Replace every number in 'ref 607 has 8 items' with '<n>'. | `"607"` | "ref <n> has 8 items" | non |
| holdout-uppercase | Replace each uppercase letter in 'AbC dEF' with '_'. | `"([A-Z])"` | "_b_ d__" | oui |
| holdout-literal-dollar | Replace every literal dollar sign in '$7 and $9' with 'USD'. | `"\\$"` | "USD7 and USD9" | oui |
| holdout-literal-star | Replace every literal asterisk in 'x*y**z' with '!'. | `".*"` | "!!" | non |
| holdout-delete-digits | Remove all digits from 'a8b90c'. | `"\\d+"` | "abc" | oui |
| holdout-whitespace-runs | Replace each run of whitespace in 'red   blue green' with '_'. | `"\\s+"` | "red_blue_green" | oui |
| holdout-literal-word | Replace every occurrence of 'fox' with 'owl' in 'fox and fox'. | `"fox"` | "owl and owl" | oui |
| holdout-explicit-pattern | Replace matches of the regex '[A-F]+' in 'tag AB and F end' with '?'. | `"[A-F]+"` | "tag ? and ? end" | oui |
| holdout-punctuation | Replace every comma in 'a,b,c' with ';'. | `","` | "a;b;c" | oui |
| holdout-leading-trailing | Replace the word 'blue' with 'red' in '  blue blue  ', preserving outer spaces. | `"blue"` | "  red red  " | oui |
| holdout-literal-brackets | Remove all literal square brackets from '[x] and [y]'. | `"[\\[\\]]+"` | "x and y" | oui |
| holdout-whole-word | Replace the whole word 'sun' with 'moon' in 'sun sunny sun'. | `"sun"` | "moon moonny moon" | non |

### Huit formulations nouvelles

| Cas | Demande | Motif produit | Résultat | Correct |
|---|---|---|---|---|
| wildcard | Use the regex 'a.*b' to replace the matched substring in 'aXXb' with '#'. | `"a.*b"` | "#" | oui |
| anchor | Replace matches of the regex '^hi' in 'hi hi' with '#'. | `"^hi"` | "# hi" | oui |
| literal-class | Use the regex '[.]' to replace dots in 'u.v.w' with '_'. | `"[.]"` | "u_v_w" | oui |
| negated-class | Remove matches of the regex '[^0-9]+' from 'a12-b3'. | `"[^0-9]+"` | "123" | oui |
| other-vowels | Replace every lowercase vowel in 'education queue' with '_'. | `"lowercase vowels"` | "education queue" | non |
| mixed-vowels | Replace all vowels in 'Audio ECHO' with '*'. | `"[^aeiouAEIOU]"` | "Au*io*E**O" | non |
| literal-operators | Remove the literal text '.*' from 'prefix .* suffix'. | `"\\.\\*"` | "prefix  suffix" | oui |
| literal-caret | Replace each literal caret in 'x^y^z' with '_'. | `"\\^"` | "x_y_z" | oui |

## Annexe E — Empreintes des fichiers conservés

- data/input/functions_definition.json : 8fc98a25310ee5ea1dfb6c07c41e71c3ea98cbe527f1232a5a564eabd277d6ba
- data/input/function_calling_tests.json : 0749fb0c4d13e9ff5a797995d9cfc0111f1ec4814bc6426842a0ec416f5099d2
- pyproject.toml : 881bb891869744ebdc2f59e8ddc7c21579e52d06ad924b56d8380d400d6f60a2
- uv.lock : a8db8a0aa490d1970566b730bd2c6dbbb9ada97dc4366287e63c1e3e9b7972a5
