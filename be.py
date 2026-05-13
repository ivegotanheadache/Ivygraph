from nltk.corpus import wordnet as wn
from itertools import permutations, combinations
from llama_cpp import Llama
import networkx as nx
import sqlite3
import logging
import wikipediaapi


logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(levelname)s: %(message)s")

llm = Llama(
    model_path="",
    verbose=False,
    n_ctx=8192,
    n_threads=30,
    n_batch=256,
)

DB_PATH = "merged.db"
ROOT = "entity.n"
LIST_OF_WORDS = ["sscnapoli", "messi"]


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
                lcs_node = max(common, key=lambda n: depths[n], default=None)
                if lcs_node is None:
                    continue
                score = 2 * depths[lcs_node] / (depths[a] + depths[b])
                if score >= threshold:
                    yield -score, a, b

    return [
        {"words": [a, b], "wp_distance": -neg_score}
        for neg_score, a, b in sorted(_scored_pairs())
    ]


def research(title, conn):
    rows = conn.execute("SELECT content FROM lines WHERE title LIKE ?", (title,)).fetchall()
    response = ' '.join([row[0] for row in rows])
    return ' '.join(response.split(".\n")[:5]) if len(response) >= 15 else ""


def create_graph(G, conn, words):
    #wiki = wikipediaapi.Wikipedia(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36", language="it")
    logging.debug("create graph: %s %s", conn, words)
    for word in words:
        logging.debug("Searching hyper for: %s", word)
        anchor_path = []
        paths = []
        try:
            for _ in range(11):
                word = word.strip().lower().replace(" ", "_")
                try:
                    s = wn.synsets(word)[0]
                    paths = s.hypernym_paths()[0]
                    paths.pop()
                    paths.append(word)
                    break
                except Exception:
                    try:
                        page = research(word, conn)
                    except Exception:
                        page = "NO DESCRIPTION FOUND"

                    role_hyper = {
                        "role": "system",
                        "content": "you are an agent IA with the task of finding a hypernym for a given word. The output must be EXCLUSIVELY AND ONLY the hypernym found, composed by ONLY ONE WORD, WITHOUT SAYING ANYTHING ELSE."
                    }
                    request = {
                        "role": "user",
                        "content": f"Find a hypernym for '{word}' based on this description (if present, else without description): {page}\nThen, classify THE WORD if it is a verb [VERB], noun [NOUN] or adjective [ADJ] or adverb [RADV] it ONLY the category found IN SHORT FORM IN PARENTESIS: NOUN, VERB, RADV, ADJ.\nThe output must be EXCLUSIVELY AND ONLY the hypernym found ONLY AND EXCLUSIVELY IN ENGLISH, a comma, and the category, WITHOUT SAYING ANYTHING ELSE.\nexample output: sport, NOUN"
                    }
                    logging.debug("request: %s", request)
                    response_hypernym = llm.create_chat_completion(
                        messages=[role_hyper, request],
                        max_tokens=20,
                        temperature=0.5,
                        stream=False
                    )
                    response_hypernym = response_hypernym["choices"][0]["message"]["content"].split(',')
                    pos_letter = response_hypernym[-1].strip()[0].lower()
                    hypernym = response_hypernym[0].strip().lower().replace(" ", "_")
                    print(hypernym, pos_letter)

                    temp_word = word + '.' + pos_letter
                    print(temp_word)
                    anchor_path.append(temp_word)
                    word = hypernym

            temp_path = []
            for i in paths:
                if not isinstance(i, str):
                    i = i.name()
                    temp_path.append('.'.join(i.split('.')[:-1]))
                else:
                    temp_path.append(i)
            temp_path = temp_path + anchor_path[::-1]
            print(temp_path)
            nx.add_path(G, temp_path)
        except Exception as e:
            logging.error(f"CREATE_GRAPH: Couldn't find a path for: {word}")


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


def add_multiple_leaves(G, child, listofwords):
    child_str = str(child)
    leafs = [n for n in G.successors(child) if G.out_degree(n) == 0]

    role = {
        "role": "system",
        "content": "you are an agent IA with the role to find around 2 new hyponyms for a given input. Hyponyms MUST be all different, but highly precise and relevant to the input AND the already present hyponyms. Hyponyms MUST be in the same context of the other given words. Hyponyms MUST be different one from each other. The output MUST be only a list of the found words separated by commas. Hyponyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input."
    }
    request = {
        "role": "user",
        "content": f"Find\n- 1 proper nouns of objects\n- ONLY if RELEVANT TO THE HYPERNYM 2 proper nouns of person\n- Consider ALSO the input to add context to the new words AND DON'T REPEAT THEM NEITHER THEIR SYNONYMS ABSOLUTELY: {listofwords}\n- The hypernym is: {child}\n->the already present hyponyms are: {leafs}"
    }

    response = llm.create_chat_completion(
        messages=[role, request],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )
    words = response["choices"][0]["message"]["content"].lower().split(',')
    logging.debug(f"FUNCTION ADD_MULTIPLE_LEAVES, \n HYPER: {child_str} \n HYPOS: {leafs} \n GENERATED: {words}")
    for i in words:
        nx.add_path(G, [child_str, i])


def find_synonyms(child):
    child_str = str(child)

    role = {
        "role": "system",
        "content": "you are an agent IA with the role to find from 1 to max 3 new synonyms for a given proper or improper noun. Synonyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input. If it is an improper noun, just find normal synonyms. (Like: house, home ...) If it is a proper noun, use instead known variations of that name (like: Messi, Lionel Messi, The goat ...) Synonyms found MUST be all different semantically, but highly precise and relevant to the input. The output MUST be only a list of the found words separated by commas."
    }
    request = {"role": "user", "content": f"FIND SYNONYMS ONLY IN THE SAME LANGUAGE OF THAT WORD: {child_str}"}

    response = llm.create_chat_completion(
        messages=[role, request],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )
    return response["choices"][0]["message"]["content"].lower().split(',')


def get_perm(comb):
    A = find_synonyms(comb["words"][0])
    B = find_synonyms(comb["words"][1])
    print(comb, A, B)
    for a in A:
        for b in B:
            yield {"combination": [str(a), str(b)], "wp_distance": comb["wp_distance"]}


def expand_tree(G, child, listofwords, extended=False, depths=None, max_depth=10):
    if depths is None:
        root = next(n for n in G.nodes if G.in_degree(n) == 0)
        depths = nx.single_source_shortest_path_length(G, root)

    if depths.get(child, 0) >= max_depth:
        return

    leafs = [n for n in G.successors(child) if G.out_degree(n) == 0]
    nodes = [n for n in G.successors(child) if G.out_degree(n) != 0]

    if leafs and (not nodes or extended):
        add_multiple_leaves(G, child, listofwords)

    for node in nodes:
        expand_tree(G, node, listofwords, extended=extended, depths=depths, max_depth=max_depth)


def print_nx_tree(G, node, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    line = prefix + connector + str(node) + "\n"
    children = sorted(G.successors(node))
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        extension = "    " if is_last else "│   "
        line += print_nx_tree(G, child, prefix + extension, is_last_child)
    return line


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
                if len(word) > 7:
                    yield word


def new(p, num_min=1, num_max=100, step=1):
    yield p
    for j in range(num_min, num_max, step):
        yield p + str(j)


if __name__ == "__main__":
    G = nx.DiGraph()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    create_graph(G, conn, LIST_OF_WORDS)
    expand_tree(G, ROOT, LIST_OF_WORDS)
    print(print_nx_tree(G, ROOT))

    try:
        with open("temp.txt", 'w') as q:
            for i in find_similar_leaves(G, ROOT, min_depth=1):
                for j in variances(list(get_perm(i))):
                    q.write(j + '\n')
    except Exception as e:
        logging.fatal(e)
    finally:
        del llm
