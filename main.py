import logging
import networkx as nx

from config import ROOT, LIST_OF_WORDS, EXPAND_MAX_DEPTH
from llm import llm  # noqa: F401
from graph import *
from metrics import find_similar_leaves
from combinations import variances, get_perm
from dedup import dedupe_file


if __name__ == "__main__":
    logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(levelname)s: %(message)s", encoding="utf-8", filemode="w")
    
    G = nx.DiGraph()

    create_graph(G, words=LIST_OF_WORDS)
    print(print_nx_tree(G, ROOT))
    expand_tree(G, child=ROOT, max_depth=EXPAND_MAX_DEPTH)
    print(print_nx_tree(G, ROOT))

    try:
        with open("wordlist.txt", 'w', encoding='utf-8') as w, open("combinations.txt", 'w', encoding='utf-8') as q:
            for i in find_similar_leaves(G, ROOT, min_depth=1):
                for j in variances(list(get_perm(i))):
                    w.write(j + '\n')
                q.write(str(i) + '\n')

    except Exception as E:
        logging.fatal("Error occurred: %s", E, exc_info=True)
    
    dedupe_file("wordlist.txt", "wordlist_deduped.txt")