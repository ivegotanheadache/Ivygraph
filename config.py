MAX_HYPONYMS = 1
MAX_SYNONYMS = 2
MAX_QUERY_ITERATIONS = 5
MIN_CHAR_TOTAL = 7
MAX_CHAR_TOTAL = 12
EXPAND_MAX_DEPTH = 15
EXTENDED = True


BACKEND = "openai"  # "openai" \ "llama"
LOCAL_LLAMA_PATH = "models/llama-2-7b-chat.gguf.q4_0.bin"  
ROOT = "entity.n"             
LIST_OF_WORDS = ["rabbit", "cyberpunk2077"] 

LINKS_FILE_PATH = "graph_links.txt" 
WORDLIST_FILE_PATH = "wordlist.txt"
COMBINATIONS_FILE_PATH = "combinations.txt"