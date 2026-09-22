import heapq
import os
import tempfile


def dedupe_file(input_path: str, output_path: str, chunk_size_lines: int = 100_000) -> None:
    """ External Merge Sort"""
    temp_files = []

    try:
        # 1. Lettura a blocchi (chunk), ordinamento in RAM e salvataggio su disco
        with open(input_path, "r", encoding="utf-8") as f:
            chunk = []
            for line in f:
                chunk.append(line)
                if len(chunk) >= chunk_size_lines:
                    chunk.sort()
                    t = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
                    t.writelines(chunk)
                    t.close()
                    temp_files.append(t.name)
                    chunk.clear()

            # Processa l'ultimo blocco rimanente
            if chunk:
                chunk.sort()
                t = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
                t.writelines(chunk)
                t.close()
                temp_files.append(t.name)

        # 2. Apertura di tutti i file temporanei
        file_handles = [open(tf, "r", encoding="utf-8") for tf in temp_files]

        # 3. K-way Merge in streaming con heapq.merge e deduplicazione adiacente
        with open(output_path, "w", encoding="utf-8") as out:
            prev = None
            # heapq.merge legge riga per riga da ciascun file in modo efficiente
            for line in heapq.merge(*file_handles):
                if line != prev:
                    out.write(line)
                    prev = line

        # Chiusura degli handle
        for fh in file_handles:
            fh.close()

    finally:
        # Pulizia dei file temporanei creati su disco
        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)