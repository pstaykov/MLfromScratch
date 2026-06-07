from pathlib import Path
import nltk

text = Path(__file__).with_name("shakespeare_gutenberg.txt").read_text(encoding="utf-8")
tokens = nltk.word_tokenize(text.lower())

n = 4
ngrams = list(nltk.ngrams(tokens, n))

model = nltk.ConditionalFreqDist((tuple(g[:n-1]), g[n-1]) for g in ngrams)

s = ["It", "is", "the", "law"]
s = [w.lower() for w in s]

for _ in range(50):
    context = tuple(s[-(n-1):])
    if context in model:
        next_word = model[context].max()
        s.append(next_word)
    else:
        break

print(" ".join(s))