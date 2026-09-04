import re
import ast

INPUT_FILE = "risultato.txt"
OUTPUT_FILE = "path_complete.txt"

def extract_path(line):
    """Estrae la lista dalla riga del log, es: ['entity.n', 'physical_entity.n', ...]"""
    match = re.search(r"\[.*?\]", line)
    if match:
        try:
            return ast.literal_eval(match.group())
        except:
            return None
    return None

with open(INPUT_FILE, "r", encoding="utf-16") as f:
    lines = f.readlines()

results = []

for i, line in enumerate(lines):
    current_path = extract_path(line)
    if current_path is None:
        continue

    # Guarda la riga successiva
    if i + 1 < len(lines):
        next_path = extract_path(lines[i + 1])
        # Se il path corrente è prefisso del prossimo → è incompleto, skip
        if next_path and next_path[:len(current_path)] == current_path:
            continue

    # Altrimenti è il path completo → tienilo
    results.append(str(current_path))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(r + "\n")

print(f"Estratte {len(results)} path complete → {OUTPUT_FILE}")