"""Re-download the external benchmark data into data/ (gitignored).

Sources (standard research mirrors):
  SimLex-999, WordSim-353, MEN  — mfaruqui/eval-word-vectors (word1 word2 score)
  Warriner et al. 2013 valence norms — JULIELab/XANEW (original ratings CSV)
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FILES = {
    "simlex.txt": "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-SIMLEX-999.txt",
    "ws353.txt": "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-WS-353-ALL.txt",
    "men.txt": "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-MEN-TR-3k.txt",
    "warriner.csv": "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv",
}

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            print(f"fetching {name}")
            urllib.request.urlretrieve(url, path)
    print("data ready")
