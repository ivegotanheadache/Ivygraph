# IvyGraph
> A semantic graph-based wordlist generator powered by WordNet, Wikipedia and LLMs.

IvyGraph builds a **semantic knowledge graph** starting from a small set of keywords, expands it using an AI agent, and generates contextually relevant word combinations applicable to any domain requiring semantic vocabulary expansion, like fuzzing, query expansion, generation of coherent and specific vocabulary, tagging...

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
```
"football", "messi", "champions league", "tennis", "formula1", "verstappen"
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
it completely excludes the combination between the two due to semantic incorrelation (See the `poc/` folder), 
without sloppy outputs that a simple and uncontrolled response to prompt will have.
With the graph you potentially can also control how much "expand" the depth of a word (leaf), by adding to it
new hyponyms. That would obviously consist in a more complex graph, I'm working on personalizing those specifities.

---

## Setup
### Prerequisites:
Python 3.14.5

### 1. Clone the repository
```bash
git clone https://github.com/ivegotanheadache/Ivygraph.git
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
python -c "import nltk; nltk.download('wordnet')"
```

For local LLM support (optional):
```bash
pip install llama-cpp-python
```

### 4. Configure your API key and config.py
Create a `.env` file in the project root:
OPENAI_API_KEY=your_key_here

IvyGraph also supports a **local LLM via llama.cpp** as an alternative to OpenAI — uncomment the `Llama` block in the source and set your model path.

Then configure the `config.py` file with you settings

```
MAX_HYPONYMS = 5   # Maximum number of hyponyms to add to the graph for each leaf
MAX_SYNONYMS = 2   # Maximum number of synonyms to consider for each combination of words
MAX_QUERY_ITERATIONS = 11   # Maximum number of iterations for each wikipedia/llm query
MIN_CHAR_TOTAL = 0   # Minimum total number of characters for outputs final result
MAX_CHAR_TOTAL = 12   # Maximum total number of characters outputs final result
EXPAND_MAX_DEPTH = 15   # Maximum depth to expand the graph
EXTENDED = True   # Whether to use extended graph expansion


BACKEND = "openai"  # "openai" \ "llama"
LOCAL_LLAMA_PATH = ""  
ROOT = "entity.n"          #DO NOT CHANGE THIS VALUE, IT IS THE ROOT OF THE GRAPH     
LIST_OF_WORDS = []         #LIST OF INPUT WORDS
LINKS_FILE_PATH = "files/graph_links.txt" 
WORDLIST_FILE_PATH = "files/wordlist.txt"
COMBINATIONS_FILE_PATH = "files/combinations.txt"
```
### 5. Run

Run the main.py
`python3 .\main.py`



---

## Disclaimer

IvyGraph is intended for **educational, research and authorized security testing only**.
Only use it against systems you own or have explicit permission to test.
