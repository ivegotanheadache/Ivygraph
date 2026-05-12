from nltk.corpus import wordnet as wn
from itertools import permutations
from llama_cpp import Llama
import networkx as nx
import matplotlib.pyplot as plt
import sqlite3
import logging


logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(levelname)s: %(message)s")

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

def find_similar_leaves(G, root, threshold=0.7, min_depth=2):
    leaves     = get_leaves(G)
    depths     = nx.single_source_shortest_path_length(G, root)
    heights    = compute_heights(G)

    # --- ancestor_to_leaves: invariato, è struttura necessaria ---
    ancestor_to_leaves = {}
    for leaf in leaves:
        for ancestor in nx.ancestors(G, leaf):
            if heights[ancestor] >= min_depth:
                ancestor_to_leaves.setdefault(ancestor, set()).add(leaf)

    # --- genera, deduplicà e punteggia in un solo passaggio ---
    seen = set()

    def _scored_pairs():
        for grouped_leaves in ancestor_to_leaves.values():
            if len(grouped_leaves) < 2:
                continue
            for a, b in combinations(sorted(grouped_leaves), 2):
                if (a, b) in seen:
                    continue
                seen.add((a, b))

                # interseca iterando il più piccolo, evita due set completi
                ancs_a = nx.ancestors(G, a)
                ancs_b = nx.ancestors(G, b)
                small, large = (ancs_a, ancs_b) if len(ancs_a) <= len(ancs_b) else (ancs_b, ancs_a)
                common = (n for n in small if n in large)

                lcs_node = max(common, key=lambda n: depths[n], default=None)
                if lcs_node is None:
                    continue

                score = 2 * depths[lcs_node] / (depths[a] + depths[b])
                if score >= threshold:
                    yield -score, a, b   # negato per sorted() ascendente

    # sorted() consuma il generatore senza mai tenere *results* separati
    return [
        {"words": [a, b], "wp_distance": -neg_score}
        for neg_score, a, b in sorted(_scored_pairs())
    ]


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
        paths = []
        try:
            for _ in range(0,11):
                word = word.strip().lower().replace(" ", "_")
                try:
                    s = wn.synsets(word)[0]

                    paths = s.hypernym_paths()[0]  
                    logging.debug(f"before path {paths}")
                    paths.pop()
                    paths.append(word) 
                    logging.debug(f"after path {paths}, {word}")    
                    break
                
                except Exception as e:
                    #La funzione di estrazione di titolo e summary dovrebbe essere implementata offline 
                    #wiki = wikipediaapi.Wikipedia(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36", language="it")
                    try:
                        page = research(word, conn)
                        logging.debug("response wiki: %s %s", word, page)
                    except Exception as e:
                        page = "NO DESCRIPTION FOUND"

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
            for i in  paths:
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

def add_multiple_leaves(G,child, listofwords):
    child_str = str(child)
    leafs = [n for n in G.successors(child) if G.out_degree(n)==0]
    role = {"role": "system", 
            "content": """ you are an agent IA with the role to find around 2 new hyponyms for a given input.
            Hyponyms MUST be all different, but highly precise and relevant to the input AND the already present hyponyms. 
            Hyponyms MUST be in the same context of the other given words.
            Hyponyms MUST be different one from each other.
            The output MUST be only a list of the found words separated by commas. 
            Hyponyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input."""}

    request = {"role": "user", "content": f"""Find
               - 1 proper nouns of objects
               - ONLY if RELEVANT TO THE HYPERNYM 2 proper nouns of pearson
               - Consider ALSO the input to add context to the new words AND DON'T REPEAT THEM NEITHER THEIR SYNONYMS ABSOLUTELY: {listofwords} 
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
    words=response.lower().split(',')
    logging.debug(f"FUNCTION ADD_MULTIPLE_LEAVES, \n HYPER: {child_str} \n HYPOS:  {leafs} \n GENERATED: {words}")
    for i in words:
        nx.add_path(G, [child_str, i])
        
def find_synonyms(child):
    child_str = str(child)
    role = {"role": "system", 
            "content": """ you are an agent IA with the role to find from 1 to max 3 new synonyms for a given proper or improper noun.
                            Synonyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input.
                            If it is a improper noun, just find normla synonyms. (Like: house, home ...)
                            If it is an proper noun, use instead known variations of that name (like: Messi, Lionel Messi, The goat ...)
                            Synonyms found MUST be all different semantically, but highly precise and relevant to the input. 
                            The output MUST be only a list of the found words separated by commas. 
                            
                            """}

    request = {"role": "user", "content": f"""FIND SYNONYMS ONLY IN THE SAME LANGUAGE OF THAT WORD : {child_str} """}
            

    response = llm.create_chat_completion(
        messages=[role, request],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )
    response = response["choices"][0]["message"]["content"]
    words=response.lower().split(',')
    return words

def get_perm(comb):
    A= find_synonyms(comb["words"][0])
    B = find_synonyms(comb["words"][1])
    print(comb, A, B)
    for a in A:
        for b in B:
            yield {"combination":[str(a),str(b)], "wp_distance":comb["wp_distance"]}
    
def expand_tree(G, child, listofwords, extended=False, depths=None, max_depth=10):
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

def to_leet(word, wovels=True, letters=True):
    l_map = {
        's': '5', 't': '7', 'b': '8', 'g': '9',
        'l': '1', 'z': '2'
    }
    w_map = {'a': '4', 'e': '3', 'i': '1', 'o': '0'}
    if wovels and letters:
        map = l_map.update(w_map)
    elif not letters:
        map = w_map
    else:
        map = l_map #se sono entrambe false gestisce l'errore come default case
        
def variances(combs, skip_short=True, skip_long=False):
    for combination in combs:
        sett = combination['combination']

        # Costruzione alphabet in un solo passaggio, senza liste temporanee
        alphabet = [
            word
            for i in sett
            for word in i.split(' ')
            if (not skip_short or len(word) > 2)
            and (not skip_long or len(word) < 11)
        ]

        # yield invece di accumulare in una lista
        for j in range(1, len(alphabet) + 1):
            for p in permutations(alphabet, j):
                word = ''.join(p).capitalize()
                if len(word) > 7:
                    yield word

def new(p, num_min, num_max, step):
    yield p
    yield to_leet(p)
    for j in range(num_min, num_max, step):
        yield p + str(j)

            
            
                

#----------------------#
if __name__== "__main__":
    
    
    G = nx.DiGraph()
    DB_PATH = "merged.db"
    ROOT = "entity.n" #default for wordnet 
    #list_of_words = ["annachiaragargiulo", "SSCNapoli","tabacchigargiulo","tigabelas_pizzeria","sorrentocalcio1945", "stanleylobotka37","luigidimaio","movimento5stelle","kkoulibaly26","driesmertens"]
    list_of_words=["boca juniors","buenos aires","napoli"]
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON") 
    create_graph(G, conn, list_of_words)
    expand_tree(G, ROOT, list_of_words)
    print(print_nx_tree(G, ROOT))
    """
   
    G = nx.DiGraph()
    ROOT = "a"
    nx.add_path(G, ("a", "Lionel Messi"))
    nx.add_path(G, ("a", "Cane")) 
    """
    try: 
        
        with open("temp.txt", 'w') as q:
            for i in find_similar_leaves(G, ROOT, min_depth = 1): 
                for j in variances(list(get_perm(i))):
                    q.write(j+'\n')
        
        
        text = print_nx_tree(G, ROOT)
    except Exception as E:
        logging.fatal(E)
    finally:    
        del llm
    """
    with open("wordlist.txt", "w") as f, open("temp.txt", 'r') as q:
        for perm in q:
            perm = perm.rstrip('\n')
            for p in new(perm, 1,100,1):
                try:
                    f.write(p + '\n')
                except Exception as E:
                    logging.error(f"{E} , word: {p}")
                    pass"""