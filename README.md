# IvyGraph
> A semantic graph-based wordlist generator powered by WordNet, Wikipedia and LLMs.

IvyGraph builds a **semantic knowledge graph** starting from a small set of keywords, expands it using an AI agent, and generates contextually relevant word combinations applicable to any domain requiring semantic vocabulary expansion
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
