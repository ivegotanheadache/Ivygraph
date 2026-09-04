import logging
import time
import re
import ast
import networkx as nx
from nltk.corpus import wordnet as wn

from config import MAX_QUERY_ITERATIONS, EXPAND_MAX_DEPTH, EXTENDED
from llm import llm, HypernymResponse, HyponymsResponse, SynonymsResponse

# Mappatura tra le categorie grammaticali restituite dall'LLM (schema HypernymResponse)
# e le lettere di categoria usate da WordNet. ADV -> 'r' (non 'a', che è già ADJ).
POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def create_graph(G, words, wiki=None):
    logging.debug(f"CREATE_GRAPH words: {words}")
    for original_word in words:
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
                    The output must be EXCLUSIVELY AND ONLY the hypernym found, composed by ONLY ONE WORD.
                    You will have a description and a list of words that describe contexts.
                    If context and description have no correlation, find an hypernym for the meaning of that word in that context.
                    Also classify the word as NOUN, VERB, ADJ or ADV.
                    """}

                    request = {"role": "user", "content": f"""Find a hypernym for '{word}'
                                based in this context: {words}
                                based on this description (if present, else without descrition): {summary}
                                Then classify THE WORD as NOUN, VERB, ADJ or ADV."""}
                    logging.debug("request: %s", request)

                    try:
                        # Chiamata strutturata: la risposta è già validata contro
                        # lo schema HypernymResponse (hypernym: str, pos: Literal[...]),
                        # niente più parsing testuale con .split(',').
                        result = llm.create_structured_completion(
                            messages=[role_hyper, request],
                            schema=HypernymResponse,
                            max_tokens=20,
                            temperature=0.5,
                        )
                        hypernym = result.hypernym.strip().lower().replace(" ", "_")
                        pos_letter = POS_MAP[result.pos]
                    except Exception as parse_e:
                        # FIX: prima un output LLM malformato veniva inghiottito
                        # dall'except più esterno senza un log specifico.
                        logging.error(f"CREATE_GRAPH: risposta LLM non valida per '{word}': {parse_e}")
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
            Hyponyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input."""}

    request = {"role": "user", "content": f"""Find ONLY IF POSSIBLE FOR EACH ONE:
               - 1 to 4 proper nouns of objects
               - ONLY if RELEVANT TO THE HYPERNYM one proper nouns of pearson
               - ONLY if RELEVANT TO THE HYPERNYM one proper nouns of places
               -The hypernym is: {child}
                    ->the already present hyponyms are: {leafs}
               """}

    # Chiamata strutturata: la risposta è già una lista di stringhe validata
    # (schema HyponymsResponse), niente più parsing testuale con .split(',').
    result = llm.create_structured_completion(
        messages=[role, request],
        schema=HyponymsResponse,
        max_tokens=100,
        temperature=0.7,
    )

    # FIX: pulizia dei nuovi nodi (spazi, maiuscole, entry vuote) per evitare
    # duplicati "invisibili" tipo "apple" vs " Apple".
    words = [
        w.strip().lower().replace(" ", "_")
        for w in result.words
        if w.strip()
    ]

    logging.debug(f"FUNCTION ADD_MULTIPLE_LEAVES, \n HYPER: {child_str} \n HYPOS:  {leafs} \n GENERATED: {words}")
    for word in words:
        nx.add_path(G, [child_str, word])


def expand_tree(G, child, extended=EXTENDED, depths=None, max_depth=10):

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

                            """}

    request = {"role": "user", "content": f"""FIND SYNONYMS ONLY IN THE SAME LANGUAGE OF THAT WORD : {child_str} """}

    # Chiamata strutturata: la risposta è già una lista di stringhe validata
    # (schema SynonymsResponse), niente più parsing testuale con .split(',').
    result = llm.create_structured_completion(
        messages=[role, request],
        schema=SynonymsResponse,
        max_tokens=100,
        temperature=0.7,
    )

    words = [w.strip().lower() for w in result.words if w.strip()]
    return words


def clean_node_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"^[\[\('\"]+|[\]\)'\"]+$", "", name)
    name = name.replace(" ", "_")
    return name


def clean_names_graph(G):
    mapping = {n: clean_node_name(n) for n in G.nodes if n != clean_node_name(n)}
    return nx.relabel_nodes(G, mapping)

def clean_graph(G):
    G = clean_names_graph(G)
    for ciclo in nx.simple_cycles(G):
        G.remove_edge(ciclo[-1], ciclo[0])
    return G    


def temp():
    with open('path_complete.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # salta righe vuote
                yield ast.literal_eval(line)