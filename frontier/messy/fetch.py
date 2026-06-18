"""
fetch.py — acquire REAL messy prose for the messy-text test.

Pulls live Wikipedia summaries (real, paraphrased, implicit prose) for a list of
titles -> messy.jsonl {title, text}. This is the "messy" the clean rendered facts
never had. No API key (Wikipedia REST is public).

Run:  python fetch.py --titles titles.txt --n 500 --out messy.jsonl
      python fetch.py --n 50            # uses a built-in sample if no titles file
"""

import json, time, argparse, urllib.request, urllib.parse

SAMPLE = ["Caffeine", "Aspirin", "Sodium chloride", "Glucose", "Penicillin",
          "Ethanol", "Benzene", "Insulin", "Chlorophyll", "Ozone", "Graphite",
          "Ammonia", "Acetic acid", "Dopamine", "Serotonin", "Methane",
          "Sulfuric acid", "Cholesterol", "Adrenaline", "Citric acid"]


def fetch_summary(title):
    t = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{t}"
    req = urllib.request.Request(url, headers={"User-Agent": "ada-messy-test/0.1"})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    if d.get("type", "").endswith("disambiguation"):
        return None
    return d.get("extract") or None


def main(titles_path, n, out):
    if titles_path:
        titles = [l.strip() for l in open(titles_path) if l.strip()][:n]
    else:
        titles = SAMPLE[:n]
    ok = 0
    with open(out, "w") as fo:
        for t in titles:
            try:
                text = fetch_summary(t)
            except Exception as e:
                text = None
            if text and len(text) > 80:
                fo.write(json.dumps({"title": t, "text": text}) + "\n"); ok += 1
            time.sleep(0.1)  # be polite
    print(f"wrote {ok}/{len(titles)} messy summaries -> {out}")
    if ok:
        ex = json.loads(open(out).readline())
        print("sample:", ex["title"], "->", ex["text"][:200])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="messy.jsonl")
    a = ap.parse_args()
    main(a.titles, a.n, a.out)
