from nltk.corpus import wordnet as wn
from itertools import permutations, combinations
#from llama_cpp import Llama
import networkx as nx
import logging
import wikipediaapi
from openai import OpenAI
import re
import sys
import ast
import time
from dotenv import load_dotenv
load_dotenv()

import os
APIKEY = os.getenv("OPENAI_API_KEY")

MIN = 7 #minimun lenght for words combination s
MAX = 18 #maximum lenght combination
MAIN_LANGUAGE = "en"
ROOT = "entity.n"
LIST_OF_WORDS = ["rabbit", "cyberpunk2077"]  # example


class Open_AI:
    def __init__(self, apik="", **kwargs):
        if not apik:
            logging.warning("Open_AI: nessuna API key fornita (OPENAI_API_KEY non impostata?)")
        self.client = OpenAI(api_key=apik)

    def create_chat_completion(self, messages, max_tokens=100, temperature=0.7, stream=False, **kwargs):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
        return {
            "choices": [{"message": {"content": response.choices[0].message.content}}]
        }


llm = Open_AI(apik=APIKEY)
"""


llm = Llama(
    model_path="",
    verbose=False,
    n_ctx=4096,
    n_threads=10,
    n_batch=128,
)
"""

logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(levelname)s: %(message)s", encoding="utf-8")
#----------#
#METRICS

def get_leaves(G):
    return [n for n in G.nodes if G.out_degree(n) == 0]


def compute_heights(G):
    heights = {}
    for node in reversed(list(nx.topological_sort(G))):
        children = list(G.successors(node))
        heights[node] = 1 if not children else 1 + max(heights[c] for c in children)
    return heights


def find_similar_leaves(G, root, threshold=0.7, min_depth=2):
    leaves = get_leaves(G)
    depths = nx.single_source_shortest_path_length(G, root)
    heights = compute_heights(G)

    ancestor_to_leaves = {}
    for leaf in leaves:
        for ancestor in nx.ancestors(G, leaf):
            if heights[ancestor] >= min_depth:
                ancestor_to_leaves.setdefault(ancestor, set()).add(leaf)

    seen = set()

    def _scored_pairs():
        for grouped_leaves in ancestor_to_leaves.values():
            if len(grouped_leaves) < 2:
                continue
            for a, b in combinations(sorted(grouped_leaves), 2):
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                ancs_a = nx.ancestors(G, a)
                ancs_b = nx.ancestors(G, b)
                small, large = (ancs_a, ancs_b) if len(ancs_a) <= len(ancs_b) else (ancs_b, ancs_a)
                common = (n for n in small if n in large)
                lcs_node = max(common, key=lambda n: depths.get(n, -1), default=None)
                if lcs_node is None:
                    # FIX: era un errore di sintassi (continue non indentato),
                    # bloccava l'esecuzione dell'intero script.
                    continue
                try:
                    score = 2 * depths[lcs_node] / (depths[a] + depths[b])
                except Exception as e:
                    logging.debug(f"FIND_SIMILAR_LEAVES: impossibile calcolare lo score per ({a},{b}): {e}")
                    score = 0
                if score >= threshold:
                    yield -score, a, b

    return [
        {"words": [a.split('.')[0], b.split('.')[0]], "wp_distance": -neg_score}
        for neg_score, a, b in sorted(_scored_pairs())
    ]
#----------#
#Graph functions

def create_graph(G, words, wiki=None):
    logging.debug(f"CREATE_GRAPH words: {words}")
    for original_word in words:
        # FIX: prima il ciclo esterno e quello interno condividevano la stessa
        # variabile 'word', quindi dopo il primo tentativo LLM il valore
        # originale andava perso (anche nei log di errore a fine funzione).
        word = original_word.strip().lower().replace(" ", "_")
        logging.debug(f"CREATE_GRAPH Searching hyper for: {word}")
        anchor_path = []
        paths = []
        try:
            for _ in range(0, MAX_QUERY_ITERATIONS):
                word = word.strip().lower().replace(" ", "_")
                try:
                    candidate_synsets = wn.synsets(word)
                    synset = candidate_synsets[0]
                    paths = synset.hypernym_paths()[0]
                    logging.debug(f"->before path {paths}")
                    paths.pop()
                    paths.append(word)
                    logging.debug(f"->after path {paths}, {word}")
                    break

                except Exception as e:
                    logging.warning(f"CREATE_GRAPH: No synsets found for {word}, trying to find hypernym with LLM. Error: {e}")
                    summary = "NO DESCRIPTION FOUND"
                    try:
                        if wiki:
                            page = wiki.page(word)
                            summary = page.summary[0:60]
                            logging.debug(f"CREATE_GRAPH: page for {word}, status: {page.exists()}, summary: {summary}")
                    except Exception as wiki_e:
                        logging.debug(f"CREATE_GRAPH: Wikipedia lookup failed for {word}: {wiki_e}")
                        summary = "NO DESCRIPTION FOUND"

                    role_hyper = {"role": "system", "content": """ you are an agent IA with the task of finding a hypernym for a given word.
                    The output must be EXCLUSIVELY AND ONLY the hypernym found, compose dby ONLY ONE WORD, WITHOUT SAYING ANYTHING ELSE.
                    You will have a description and a list of words that describe contexts.
                    If context and description have no correlation, find an hypernym for the meaning of that word in that context
                    """}

                    request = {"role": "user", "content": f"""Find a hypernym for '{word}'
                                based in this context: {words}
                                based on this description (if present, else without descrition): {summary}
                                Then classify THE WORD if it is a verb [VERB], noun [NOUN] or adjective [ADJ] or adverb[RADV]  it ONLY the category found IN SHORT FORM IN PARENTESIS: NOUN, VERB , RADV, ADJ.
                                The output must be EXCLUSIVELY AND ONLY the hypernym found ONLY AND EXCLUSIVELY IN ENGLISH, a comma, and the category, WITHOUT SAYING ANYTHING ELSE.
                                example output: sport, NOUN"""}
                    logging.debug("request: %s", request)

                    try:
                        response_hypernym = llm.create_chat_completion(
                            messages=[role_hyper, request],
                            max_tokens=20,
                            temperature=0.5,
                            stream=False
                        )
                        response_hypernym = response_hypernym["choices"][0]["message"]["content"].split(',')
                        pos_letter = response_hypernym[-1].strip(' ')[0].lower()
                        hypernym = response_hypernym[0].strip().lower().replace(" ", "_")
                    except (IndexError, KeyError, TypeError) as parse_e:
                        # FIX: prima un output LLM malformato veniva inghiottito
                        # dall'except più esterno senza un log specifico.
                        logging.error(f"CREATE_GRAPH: risposta LLM malformata per '{word}': {parse_e}")
                        raise

                    temp_word = word + '.' + pos_letter
                    anchor_path.append(temp_word)
                    word = hypernym
                    print(f"Trying to find hypernym for {temp_word}, got: {hypernym} with pos: {pos_letter}")

            temp_path = []
            for i in paths:
                if not isinstance(i, str):
                    i = i.name()
                    temp_path.append('.'.join(i.split('.')[0:-1]))
                else:
                    temp_path.append(i)
                logging.debug(f"{temp_path}, {i}")
            temp_path = temp_path + anchor_path[::-1]
            print(temp_path)
            nx.add_path(G, temp_path)
        except Exception as E:
            # FIX: ora logga la parola originale, non l'ultimo iperonimo tentato.
            logging.error(f"CREATE_GRAPH: Couldn't find a path for: {original_word} ({E})")

def substitute_leaf(G, old_leaf, new_leaf):
    neighbors = list(G.neighbors(old_leaf))
    if not neighbors:
        return G
    neighbor = neighbors[0]
    G.add_node(new_leaf)
    nx.set_node_attributes(G, {new_leaf: G.nodes[old_leaf]})
    G.add_edge(neighbor, new_leaf)
    G.remove_node(old_leaf)
    return G

def add_multiple_leaves(G, child):
    child_str = str(child)
    leafs = [n for n in G.successors(child) if G.out_degree(n) == 0]
    role = {"role": "system",
            "content": """ you are an agent IA with the role to find around 4 to 10 new hyponyms for a given input.
            Hyponyms MUST be all different, but highly precise and relevant to the input AND the already present hyponyms.
            Hyponyms MUST be in the same context of the other given words.
            Hyponyms MUST be different one from each other.
            The output MUST be only a list of the found words separated by commas.
            Hyponyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input."""}

    request = {"role": "user", "content": f"""Find ONLY IF POSSIBLE FOR EACH ONE:
               - 1 to 4 proper nouns of objects
               - ONLY if RELEVANT TO THE HYPERNYM one proper nouns of pearson
               - ONLY if RELEVANT TO THE HYPERNYM one proper nouns of places
               -The hypernym is: {child}
                    ->the already present hyponyms are: {leafs}
               """}

    response = llm.create_chat_completion(
        messages=[role, request],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )
    response = response["choices"][0]["message"]["content"]

    # FIX: prima i nuovi nodi non venivano puliti (spazi, maiuscole, entry vuote),
    # creando duplicati "invisibili" tipo "apple" vs " Apple".
    words = [
        w.strip().lower().replace(" ", "_")
        for w in response.lower().split(',')
        if w.strip()
    ]

    logging.debug(f"FUNCTION ADD_MULTIPLE_LEAVES, \n HYPER: {child_str} \n HYPOS:  {leafs} \n GENERATED: {words}")
    for word in words:
        nx.add_path(G, [child_str, word])

def expand_tree(G, child, extended=False, depths=None, max_depth=10):

    if depths is None:
        root = next(n for n in G.nodes if G.in_degree(n) == 0)
        depths = nx.single_source_shortest_path_length(G, root)

    if depths.get(child, 0) >= max_depth:
        return

    leafs = [n for n in G.successors(child) if G.out_degree(n) == 0]
    nodes = [n for n in G.successors(child) if G.out_degree(n) != 0]

    if leafs and (not nodes or extended):
        add_multiple_leaves(G, child)
    for node in nodes:
        time.sleep(0.1)  # Aggiungi un ritardo per evitare di sovraccaricare l'API
        expand_tree(G, node, extended=extended, depths=depths, max_depth=max_depth)

def print_nx_tree(G, node, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    line = prefix + connector + str(node) + "\n"

    children = sorted(G.successors(node))
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        extension = "    " if is_last else "│   "
        line += print_nx_tree(G, child, prefix + extension, is_last_child)

    return line

def find_synonyms(child):
    child_str = str(child)
    role = {"role": "system",
            "content": """ you are an agent IA with the role to find from 1 to max 5 new synonyms for a given proper or improper noun.
                            Synonyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input.
                            If it is a improper noun, just find normal synonyms. (Like: house, home ...)
                            If it is an proper noun, use instead known variations of that name (like: Messi, Lionel Messi, The goat ...)
                            Synonyms found MUST be all different semantically, but highly precise and relevant to the input.
                            The output MUST be only a list of the found words separated by commas AND NOTHING ELSE.
                            example: house, home, building

                            """}

    request = {"role": "user", "content": f"""FIND SYNONYMS ONLY IN THE SAME LANGUAGE OF THAT WORD : {child_str} """}

    response = llm.create_chat_completion(
        messages=[role, request],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )
    response = response["choices"][0]["message"]["content"]

    # FIX: pulizia + rimozione entry vuote, come per add_multiple_leaves.
    words = [w.strip() for w in response.lower().split(',') if w.strip()]
    return words

#------------#
#Combinations

def get_perm(comb):
    A = find_synonyms(comb["words"][0])
    B = find_synonyms(comb["words"][1])
    print(comb, A, B)
    for a in A:
        for b in B:
            yield {"combination": [str(a), str(b)], "wp_distance": comb["wp_distance"]}

def variances(combs, skip_short=True, skip_long=False):
    for combination in combs:
        sett = combination['combination']
        alphabet = [
            word
            for i in sett
            for word in i.split(' ')
            if (not skip_short or len(word) > 2)
            and (not skip_long or len(word) < 11)
        ]
        for j in range(1, len(alphabet) + 1):
            for p in permutations(alphabet, j):
                word = ''.join(p).capitalize()
                if len(word) > MIN and len(word) < MAX:
                    yield word

def new(p, num_min=1, num_max=100, step=1):
    yield p
    for j in range(num_min, num_max, step):
        yield p + str(j)

#--------
def clean_node_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"^[\[\('\"]+|[\]\)'\"]+$", "", name)
    name = name.replace(" ", "_")
    return name

def clean_graph(G):
    mapping = {n: clean_node_name(n) for n in G.nodes if n != clean_node_name(n)}
    return nx.relabel_nodes(G, mapping)

def temp():
    with open('path_complete.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # salta righe vuote
                yield ast.literal_eval(line)

if __name__ == "__main__":
    paths = []
    G = nx.DiGraph()

    for path in temp():
        nx.add_path(G, path)

    G = clean_graph(G)

    for ciclo in nx.simple_cycles(G):
        G.remove_edge(ciclo[-1], ciclo[0])

    print('concept.n' in G.nodes())

    root = [n for n in G.nodes() if G.in_degree(n) == 0]
    print(f"Root trovate: {root}")

    for r in root:
        if nx.has_path(G, r, 'concept.n'):
            print(f"Raggiungibile da {r}")
            break
    else:
        print("concept.n NON è raggiungibile da nessuna root")

    try:
        with open("wordlist.txt", 'w', encoding='utf-8') as w, open("combinations.txt", 'w', encoding='utf-8') as q:
            for i in find_similar_leaves(G, ROOT, min_depth=1):
                for j in variances(list(get_perm(i))):
                    w.write(j + '\n')
                q.write(str(i) + '\n')

    except Exception as E:
        logging.fatal("Error occurred: %s", E, exc_info=True)
    finally:
        del llm