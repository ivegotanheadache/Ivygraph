# IvyGraph
> A semantic graph-based wordlist generator powered by WordNet, Wikipedia and LLMs.

IvyGraph builds a **semantic knowledge graph** starting from a small set of keywords, expands it using an AI agent, and generates contextually relevant word combinations — applicable to any domain requiring semantic vocabulary expansion.

---

## How it works

Given a list of keywords, IvyGraph follows this pipeline:

**1. Graph construction**
Each keyword is placed in a semantic DAG by climbing its hypernym chain via WordNet.

**2. LLM fallback**
If a term is too specific or absent from WordNet, an AI agent fetches a Wikipedia summary and iteratively finds a suitable hypernym until the term can be anchored in the graph.

**3. Graph expansion**
The graph is expanded bottom-up: for each terminal node (a node with leaves), an LLM generates new contextually relevant hyponyms.

**4. Similarity scoring**
Wu-Palmer similarity is computed between all leaf pairs. The most semantically related combinations are ranked and used to generate word variants via synonym expansion and permutation.
LIST_OF_WORDS → Graph → Expanded Graph → Wu-Palmer pairs → Synonyms → Wordlist

---

## Example output

Input:
```python
LIST_OF_WORDS = ["football", "messi", "champions league", "tennis", "formula1", "verstappen"]
```

Sample generated pairs:
{'words': ['messi', 'verstappen'], 'wp_distance': 0.912}
{'words': ['football', 'tennis'], 'wp_distance': 0.876}
{'words': ['champions league', 'formula1'], 'wp_distance': 0.843}

Sample wordlist output:
Football
MessiChampions
TennisLeague
Verstappen
FormulaChampions
...

There is a Proof of Concept in the repo that's shows how, when given ["Cyberpunk2077", "Rabbit"] as input,
it completely excludes the combination between the two due to semantic incorrelation.
See the `poc/` folder.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ivygraph.git
cd ivygraph
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
OPENAI_API_KEY=your_key_here

IvyGraph also supports a **local LLM via llama.cpp** as an alternative to OpenAI — uncomment the `Llama` block in the source and set your model path.

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
MAX = 18       # maximum output word length for wordlist
ROOT = "entity.n"   # WordNet root node
```

The python file will generate three files:
   1. wordlist.txt     #the output wordlist
   2. combinations.txt #found combinations of words semantically close
   3. app.log for logs

---

## Disclaimer

IvyGraph is intended for **educational, research and authorized security testing only**.
Only use it against systems you own or have explicit permission to test.
