import platform
import subprocess


def dedupe_file(input_path: str, output_path: str) -> None:
    """
    Deduplica un file di parole (una per riga) senza caricarlo mai per
    intero in memoria. Stessa strategia su entrambe le piattaforme:
    1) ordina il file su disco (comando esterno, non in RAM Python)
    2) rimuove le righe duplicate adiacenti in streaming, tenendo in
       memoria solo l'ultima riga vista (come fa `uniq` su Unix).
    """
    if platform.system() == "Windows":
        sorted_path = input_path + ".sorted"
        # sort.exe nativo di Windows: ordina su disco, non carica tutto in RAM.
        subprocess.run(["sort", input_path, "/O", sorted_path], check=True, shell=True)
        _dedupe_adjacent(sorted_path, output_path)
    else:
        sorted_path = input_path + ".sorted"
        with open(sorted_path, "w", encoding="utf-8") as out:
            subprocess.run(["sort", input_path], stdout=out, check=True, env={"LC_ALL": "C"})
        _dedupe_adjacent(sorted_path, output_path)


def _dedupe_adjacent(sorted_path: str, output_path: str) -> None:
    with open(sorted_path, encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out:
        prev = None
        for line in f:
            if line != prev:
                out.write(line)
                prev = line