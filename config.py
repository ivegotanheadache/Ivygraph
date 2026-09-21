MAX_HYPONYMS = 5   # Maximum number of hyponyms to add to the graph for each leaf
MAX_SYNONYMS = 2   # Maximum number of synonyms to consider for each combination of words
MAX_QUERY_ITERATIONS = 11   # Maximum number of iterations for each wikipedia/llm query
MIN_CHAR_TOTAL = 0   # Minimum total number of characters for outputs final result
MAX_CHAR_TOTAL = 12   # Maximum total number of characters outputs final result
EXPAND_MAX_DEPTH = 15   # Maximum depth to expand the graph
EXTENDED = True   # Whether to use extended graph expansion


BACKEND = "openai"  # "openai" \ "llama"
LOCAL_LLAMA_PATH = "models/llama-2-7b-chat.gguf.q4_0.bin"  
ROOT = "entity.n"             
LIST_OF_WORDS = ["calcio", "linkin park"]
LINKS_FILE_PATH = "files/graph_links.txt" 
WORDLIST_FILE_PATH = "files/wordlist.txt"
COMBINATIONS_FILE_PATH = "files/combinations.txt"