# 🐝 Wonabee

> A semantic graph-based wordlist generator powered by WordNet, Wikipedia and LLMs.

Wonabee builds a **semantic knowledge graph** starting from a small set of keywords, expands it using an AI agent, and generates contextually relevant word combinations — applicable to any domain requiring semantic vocabulary expansion.

---

## How it works

Given a list of keywords (`LIST_OF_WORDS`), Wonabee follows this pipeline:

**1. Graph construction**
Each keyword is placed in a semantic DAG by climbing its hypernym chain via WordNet.

**2. LLM fallback**
If a term is too specific or absent from WordNet (e.g. `cyberpunk2077`, `stardew_valley`,`rabbit`), an AI agent fetches a Wikipedia summary and iteratively finds a suitable hypernym until the term can be anchored in the graph.

**3. Graph expansion**
The graph is expanded bottom-up: for each terminal node (a node with leaves), an LLM generates new contextually relevant hyponyms.

**4. Similarity scoring**
Wu-Palmer similarity is computed between all leaf pairs. The most semantically related combinations are ranked and used to generate word variants via synonym expansion and permutation.

```
LIST_OF_WORDS → Graph → Expanded Graph → Wu-Palmer pairs → Synonyms → Wordlist
```

---

## Example output (sports domain)

Input:
```python
LIST_OF_WORDS = ["football", "messi", "champions league", "tennis", "formula1", "verstappen"]
```

Sample generated pairs:
```
{'words': ['messi', 'verstappen'], 'wp_distance': 0.912}
{'words': ['football', 'tennis'], 'wp_distance': 0.876}
{'words': ['champions league', 'formula1'], 'wp_distance': 0.843}
```

Sample wordlist output:
```
Football
MessiChampions
TennisLeague
Verstappen
FormulaChampions
...
```

See the `poc/` folder for real output examples — including sample wordlists and combination files generated from the code.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/wonabee.git
cd wonabee
```

### 2. Create and activate a virtual environment

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For local LLM support (optional):
```bash
pip install llama-cpp-python
```

### 4. Configure your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

Wonabee also supports a **local LLM via llama.cpp** as an alternative to OpenAI — uncomment the `Llama` block in the source and set your model path.

---

## Usage

```python
# 1. Set your keywords
LIST_OF_WORDS = ["your", "keywords", "here"]

# 2. Run
python graphgen.py
```

To tune generation:
```python
MIN = 7        # minimum output word length for wordlist
MAX = 18       # maximum putput word length for wordlist
ROOT = "entity.n"   # WordNet root node
```
The python file will generate two files:
   1. wordlist.txt     #the output wordlist
   2. combinations.txt #found combinations of words semantically close

---

## Repository structure

```
wonabee/
├── graphgen.py          # main script
├── requirements.txt
├── .env                # API key (not committed)
├── .gitignore
└── poc/                # proof of concept outputs
    ├── wordlist_example.txt   # sample wordlist output 
    └── combinations_example.txt  # sample Wu-Palmer pairs and combinations
```

---

## Use cases

- 🔐 **Password wordlist generation** — given a set of interests (sport teams, hobbies, favorite artists), generate a semantically rich wordlist for use in authorized penetration testing or password auditing tools like Hashcat or John the Ripper.
  ```python
  # Example: target is a football fan
  LIST_OF_WORDS = ["messi", "barcelona", "champions league", "argentina"]
  # → generates: MessiBarca, ChampionsLeague, LionelMessi, ...
  ```

- 🧠 **Domain vocabulary extraction** — derive a structured vocabulary from a topic, useful for tagging systems, search engines or NLP preprocessing.
  ```python
  LIST_OF_WORDS = ["pizza", "mozzarella", "olive oil", "fermentation"]
  # → generates a semantic graph covering the cuisine domain
  ```

- 🗺️ **Ontology exploration** — visualize how concepts relate to each other in a given domain, leveraging WordNet's hypernym structure enriched with LLM knowledge.

- 📊 **Data augmentation** — expand a small keyword set into a broader vocabulary for training classifiers, NER models or other NLP tasks.

- 📖 **Text-based vocabulary extraction** — feed keywords extracted from a document to generate a thematic wordlist reflecting the text's semantic domain.

---

## Disclaimer

Wonabee is intended for **educational, research and authorized security testing only**.
When used for password auditing, only use it against systems you own or have explicit permission to test.
