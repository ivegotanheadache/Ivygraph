from nltk.corpus import wordnet as wn
from itertools import permutations
from llama_cpp import Llama
import networkx as nx
import matplotlib.pyplot as plt
import sqlite3
import logging


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

#----LLM setup----#

llm = Llama(
      model_path="/home/cobalt/Scrivania/models/Llama-3.3-70B-Instruct-Q4_K_L.gguf",
      verbose=False,
      n_ctx=8192,
      n_threads=30,
      n_batch=256,
)

#------------------------------------#
#Metrics#

from itertools import combinations
import networkx as nx

def get_leaves(G):
    # returns all nodes with no outgoing edges (leaf nodes)
    return [n for n in G.nodes if G.out_degree(n) == 0]

def compute_heights(G):
    """
    Per ogni nodo, calcola l'altezza = distanza dalla foglia più lontana.
    Foglie → altezza 0, loro padre → altezza 1, ecc.
    """
    heights = {}
    # processa in ordine topologico inverso (dalle foglie verso la radice)
    for node in reversed(list(nx.topological_sort(G))):
        children = list(G.successors(node))
        if not children:
            heights[node] = 1  # è una foglia
        else:
            heights[node] = 1 + max(heights[child] for child in children)
    return heights

def find_similar_leaves(G, root, threshold=0.0, min_depth=2):
    leaves = get_leaves(G)
    
    # precomputes depth of every node from root in one pass, O(n) 
    # avoids recalculating shortest_path_length for every node later
    depths = nx.single_source_shortest_path_length(G, root)
    heights=compute_heights(G)
    
    # builds a dict: ancestor -> set of leaves that have that ancestor
    # so we know which leaves are "in the same neighborhood"
    ancestor_to_leaves = {}
    for leaf in leaves:
        for ancestor in nx.ancestors(G, leaf):
            # ✅ added: skip ancestors too close to root (too generic)
            if heights[ancestor] < min_depth:
                continue
            ancestor_to_leaves.setdefault(ancestor, set()).add(leaf)

    # for each ancestor group, generate all pairs of leaves
    # using a set() to avoid computing the same pair twice
    candidates = set()
    for ancestor, grouped_leaves in ancestor_to_leaves.items():
        if len(grouped_leaves) > 1:                          # skip solo leaves
            for a, b in combinations(grouped_leaves, 2):
                candidates.add((min(a,b), max(a,b)))         # (dog,cat) == (cat,dog)

    # now compute wu-palmer ONLY for candidate pairs
    results = []
    for a, b in candidates:
        # finds the deepest common ancestor (lowest common subsumer)
        lcs_node = max(
            nx.ancestors(G, a) & nx.ancestors(G, b),         # ancestors in common
            key=lambda n: depths[n]                           # pick the deepest one
        )
        # wu-palmer formula: 2*depth(lcs) / (depth(a) + depth(b))
        score = 2 * depths[lcs_node] / (depths[a] + depths[b])
        
        if score >= threshold:                               # filter by threshold
            results.append({"words": [a, b], "wp_distance": score})

    # sort by score descending so most similar pairs come first
    return sorted(results, key=lambda x: -x["wp_distance"])




# DATABASE QUERY#


def research(title,conn):
    rows = conn.execute("SELECT content FROM lines WHERE title LIKE ?", (title,)).fetchall()
    response = ' '.join([row[0] for row in rows]) 

    if len(response) < 15:
        return ""
    else:
        return ' '.join(response.split(".\n")[:5])


#------------------------------------#
#GRAPH FUNCTIONS#

def create_graph(G,  conn, words):
    logging.debug("create graph: %s %s", conn, words)
    for word in words:
        logging.debug("Searching hyper for: %s", word)
        anchor_path=[]

        while True:
            word = word.strip().lower().replace(" ", "_")
            try:
                s = wn.synsets(word)[0]

                paths = s.hypernym_paths()        
                break
            
            except Exception as e:
                #La funzione di estrazione di titolo e summary dovrebbe essere implementata offline 
                #wiki = wikipediaapi.Wikipedia(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36", language="it")
                
                page = research(word, conn)
                logging.debug("response wiki: %s %s", word, page)
                
                role_hyper = {"role": "system", "content": """ you are an agent IA with the task of finding a hypernym for a given word. 
                The output must be EXCLUSIVELY AND ONLY the hypernym found, compose dby ONLY ONE WORD, WITHOUT SAYING ANYTHING ELSE.
                """}

                request =  {"role": "user", "content": f"""Find a hypernym for '{word}' based on this description (if present, else without descrition): {page}
                Then,classify THE WORD if it is a verb [VERB], noun [NOUN] or adjective [ADJ] or adverb[RADV]  it ONLY the category found IN SHORT FORM IN PARENTESIS: NOUN, VERB , RADV, ADJ.
                The output must be EXCLUSIVELY AND ONLY the hypernym found ONLY AND EXCLUSIVELY IN ENGLISH, a comma, and the category, WITHOUT SAYING ANYTHING ELSE.
                example output: sport, NOUN"""}
                logging.debug("request: %s", request)
                response_hypernym = llm.create_chat_completion(
                    messages=[role_hyper, request],
                    max_tokens=20,
                    temperature=0.5,
                    stream=False
                )
                
                response_hypernym = response_hypernym["choices"][0]["message"]["content"].split(',')
                #print(response_hypernym)
                pos_letter = response_hypernym[-1].strip(' ')[0].lower()
                hypernym = response_hypernym[0].strip().lower().replace(" ", "_")
                print(hypernym, pos_letter)
                
                temp_word = word + '.' + pos_letter
                
                print(temp_word)
                
                anchor_path.append(temp_word)
                word = hypernym

        
        temp_path = []        
        for i in  paths[0]:
            temp_path.append('.'.join(i.name().split('.')[0:-1]))
        temp_path = temp_path + anchor_path[::-1]
        print(temp_path)
        nx.add_path(G, temp_path)
        
def substitute_leaf(G, old_leaf, new_leaf):
    neighbor = list(G.neighbors(old_leaf))[0] 
    G.add_node(new_leaf)
    nx.set_node_attributes(G, {new_leaf: G.nodes[old_leaf]})
    G.add_edge(neighbor, new_leaf)
    G.remove_node(old_leaf)
    return G

def add_multiple_leaves(G,child):
    child_str = str(child)
    leafs = [n for n in G.successors(child) if G.out_degree(n)==0]
    role = {"role": "system", 
            "content": """ you are an agent IA with the role to find around 2 new hyponyms for a given input.
            Hyponyms MUST be all different, but highly precise and relevant to the input. 
            The output MUST be only a list of the found words separated by commas. 
            Hyponyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input."""}

    request = {"role": "user", "content": f"""Find
               - 1 proper nouns of objects
               - ONLY if RELEVANT TO THE HYPERNYM 2 proper nouns of pearson
                for this hipernyms : {child_str}. 
                Consider ALSO the already present hyponym to add context to the new words AND DON'T REPEAT THEM: {leafs} """}
            

    response = llm.create_chat_completion(
        messages=[role, request],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )
    response = response["choices"][0]["message"]["content"]
    words=response.lower().split(',')
    
    for i in words:
        nx.add_path(G, [child_str, i])
        
def find_synonyms(child):
    child_str = str(child)
    role = {"role": "system", 
            "content": """ you are an agent IA with the role to find from 1 to max 5 new synonyms for a given proper or improper noun.
                            If it is a improper noun, just find normla synonyms. (Like: house, home ...)
                            If it is an proper noun, use instead known variations of that name (like: Messi, Lionel Messi, The goat ...)
                            Synonyms found MUST be all different semantically, but highly precise and relevant to the input. 
                            The output MUST be only a list of the found words separated by commas. 
                            Improper nouns MUST be in the IN THE ORIGINAL LANGUAGE of the given input.
                            """}

    request = {"role": "user", "content": f"""Find synonyms for this word : {child_str}. """}
            

    response = llm.create_chat_completion(
        messages=[role, request],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )
    response = response["choices"][0]["message"]["content"]
    words=response.lower().split(',')
    print(response)
    return words

def get_perm(comb):
    A= find_synonyms(comb["words"][0])
    B = find_synonyms(comb["words"][1])
    print(comb, A, B)
    for a in A:
        for b in B:
            yield {"combination":[str(a),str(b)], "wp_distance":comb["wp_distance"]}
    
def expand_tree(G, child, extended=False, depths=None, max_depth=5):
    if depths is None:
        # calcolato una sola volta alla prima chiamata
        root = next(n for n in G.nodes if G.in_degree(n) == 0)
        depths = nx.single_source_shortest_path_length(G, root)

    # ferma l'espansione se child è già troppo profondo
    if depths.get(child, 0) >= max_depth:
        return

    leafs = [n for n in G.successors(child) if G.out_degree(n) == 0]
    nodes = [n for n in G.successors(child) if G.out_degree(n) != 0]

    if leafs and (not nodes or extended):
        add_multiple_leaves(G, child)

    for node in nodes:
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

def to_leet(word):
    leet_map = {
        'a': '4', 'e': '3', 'i': '1', 'o': '0',
        's': '5', 't': '7', 'b': '8', 'g': '9',
        'l': '1', 'z': '2'
    }
    return ''.join(leet_map.get(c.lower(), c) for c in word)
def variances(combs, skip_short=True, skip_long=False):
    perms = []
    for combinations in combs:
        alphabet=[]
        sett= combinations['combination']
        print(combs, sett, alphabet)
        for i in sett:
            alphabet.extend(i.split(' '))
        if skip_short:
            alphabet = [i for i in alphabet if len(i)>2]
        if skip_long:
            alphabet = [i for i in alphabet if len(i)<11]
        print(alphabet)
        for j in range(1, len(alphabet) + 1):
            perms.extend(
                word
                for p in permutations(alphabet, j)
                if len(word := ''.join(p).capitalize()) > 7
            )
    return perms

def leet_rule(perms):
    for p in perms:
        yield to_leet(p)

def number_rule(perms, num_min=0, num_max=99, step=1):
    for p in perms:
        for j in range(num_min, num_max, step):
            yield p + str(j)

            
            
                

#----------------------#
if __name__== "__main__":
    """
    G = nx.DiGraph()
    DB_PATH = "merged.db"
    ROOT = "entity.n" #default for wordnet 
    list_of_words = ["Maradona", "SSCNapoli"]
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON") 
    
    create_graph(G, conn, list_of_words)
    expand_tree(G, ROOT, ROOT, extended=True)
    text = print_nx_tree(G, "entity.n")
    print(text)
    """
    H = nx.DiGraph()
    ROOT = "a"
    nx.add_path(H, ("a", "Lionel Messi"))
    nx.add_path(H, ("a", "Cane"))
    results=[]
    with open("wordlist.txt", "w") as f:
        for i in find_similar_leaves(H, ROOT, min_depth= 1):
            #print(list(get_perm(i)))
            perms = variances(list(get_perm(i)))
        #print("PERMUTATIONS", perms)
            perms = leet_rule(perms)
            perms = number_rule(perms)
            for p in perms:
                f.write(p + "\n")