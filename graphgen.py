from nltk.corpus import wordnet as wn
from itertools import permutations, combinations
from llama_cpp import Llama
import networkx as nx
import logging
import wikipediaapi
from openai import OpenAI
import re
from dotenv import load_dotenv
load_dotenv()

import os
APIKEY = os.getenv("OPENAI_API_KEY")

MIN = 7 #minimun lenght for words combination s
MAX = 18 #maximum lenght combination 
MAIN_LANGUAGE="en"
ROOT = "entity.n"
LIST_OF_WORDS = ["rabbit", "cyberpunk2077"]


class Open_AI:
    def __init__(self, apik="", **kwargs):
        self.client = OpenAI(api_key=apik)

    def create_chat_completion(self, messages, max_tokens=100, temperature=0.7, stream=False, **kwargs):
        response = self.client.chat.completions.create(
            model="gpt-4o",
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

logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(levelname)s: %(message)s")
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
                lcs_node = max(common, key=lambda n: depths[n], default=None)
                if lcs_node is None:
                    continue
                score = 2 * depths[lcs_node] / (depths[a] + depths[b])
                if score >= threshold:
                    yield -score, a, b
    
    return [
        {"words": [a.split('.')[0], b.split('.')[0]], "wp_distance": -neg_score}
        for neg_score, a, b in sorted(_scored_pairs())
    ]
#----------#
#Graph functions

def create_graph(G, words,wiki = None):
    logging.debug(f"CREATE_GRAPH words: {words}")
    for word in words:
        logging.debug(f"CREATE_GRAPH Searching hyper for: {word}")
        anchor_path=[]
        paths = []
        try:
            for _ in range(0,11):
                word = word.strip().lower().replace(" ", "_")
                try:
                    synsets = wn.synsets(word)
                    synsets = synsets[0]   
                    paths = synsets.hypernym_paths()[0]  
                    logging.debug(f"->before path {paths}")
                    paths.pop()
                    paths.append(word) 
                    logging.debug(f"->after path {paths}, {word}")    
                    break
                
                except Exception as e:
                
                    try:
                        if wiki:
                            #wiki = wikipediaapi.Wikipedia(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36", language=MAIN_LANGUAGE) 
                            page = wiki.page(word)
                            summary = page.summary[0:60]
                            logging.debug(f"CREATE_GRAPH: page for {word}, status: {page.exists()}, summary: {summary}")
                    except Exception as e:
                        summary = "NO DESCRIPTION FOUND"

                    role_hyper = {"role": "system", "content": """ you are an agent IA with the task of finding a hypernym for a given word. 
                    The output must be EXCLUSIVELY AND ONLY the hypernym found, compose dby ONLY ONE WORD, WITHOUT SAYING ANYTHING ELSE.
                    You will have a description and a list of words that describe contexts.
                    If context and description have no correlation, find an hypernym for the meaning of that word in that context
                    """}

                    request =  {"role": "user", "content": f"""Find a hypernym for '{word}' 
                                based in this context: {words}
                                based on this description (if present, else without descrition): {summary} 
                                Then classify THE WORD if it is a verb [VERB], noun [NOUN] or adjective [ADJ] or adverb[RADV]  it ONLY the category found IN SHORT FORM IN PARENTESIS: NOUN, VERB , RADV, ADJ.
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
            "content": """ you are an agent IA with the role to find around 4 to 10 new hyponyms for a given input.
            Hyponyms MUST be all different, but highly precise and relevant to the input AND the already present hyponyms. 
            Hyponyms MUST be in the same context of the other given words.
            Hyponyms MUST be different one from each other.
            The output MUST be only a list of the found words separated by commas. 
            Hyponyms MUST be in the IN THE ORIGINAL LANGUAGE of the given input."""}

    request = {"role": "user", "content": f"""Find ONLY IF POSSIBLE FOR EACH ONE:
               - 1 to 5 proper nouns of objects
               - ONLY if RELEVANT TO THE HYPERNYM 2 to 5 proper nouns of pearson
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
    words=response.lower().split(',')
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
            
            
    #yield {"combination":[comb["words"][0],comb["words"][1]], "wp_distance":comb["wp_distance"]}

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

if __name__ == "__main__":
    
    
    wiki = wikipediaapi.Wikipedia(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36", language=MAIN_LANGUAGE) 
   
    G = nx.DiGraph()
    create_graph(G, words=LIST_OF_WORDS, wiki=wiki)
    
   
    expand_tree(G, ROOT, LIST_OF_WORDS, extended=True)
    G = clean_graph(G)
    print(print_nx_tree(G, ROOT))
    
    try: 
        with open("wordlist.txt", 'w') as w, open("combinations.txt", 'w') as q:
            for i in find_similar_leaves(G, ROOT, min_depth = 1): 
                
                for j in variances(list(get_perm(i))): 
                    w.write(j+'\n')
                q.write(str(i)+'\n')
            text = print_nx_tree(G, ROOT)
            q.write(text)
            
    except Exception as E:
        logging.fatal(E)
    finally:    
        del llm
