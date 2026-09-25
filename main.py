import logging
import networkx as nx
import wikipediaapi

from config import ROOT, LIST_OF_WORDS, WORDLIST_DEDUPLICATED_FILE_PATH, WORDLIST_FILE_PATH, COMBINATIONS_FILE_PATH
from llm import get_llm
from graph import *
from metrics import find_similar_leaves
from combinations import variances, get_perm
from dedup import dedupe_file


if __name__ == "__main__":
    logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(levelname)s: %(message)s", encoding="utf-8", filemode="w")
    wiki = wikipediaapi.Wikipedia(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", language='en')
    llm = get_llm()
    
    G = nx.DiGraph()
    create_graph(G, words=LIST_OF_WORDS, wiki=wiki)
    G = clean_graph(G, ROOT)
    expand_tree(G, ROOT)
    print(print_nx_tree(G, ROOT))
    try:
        with open(WORDLIST_FILE_PATH, 'w', encoding='utf-8') as w, open(COMBINATIONS_FILE_PATH, 'w', encoding='utf-8') as q:
            for combination in find_similar_leaves(G, ROOT, min_depth=1):
                for j in variances(get_perm(combination["words"]), skip_short=True, skip_long=True):
                    w.write(j + '\n')
                q.write(str(combination) + '\n')
    
    except Exception as E:
        logging.fatal("Error occurred: %s", E, exc_info=True)
    
    dedupe_file(WORDLIST_FILE_PATH, WORDLIST_DEDUPLICATED_FILE_PATH)