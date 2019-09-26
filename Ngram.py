import nltk
import nltk.lm
import nltk.corpus
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.util import ngrams, everygrams
from nltk.lm import MLE

mobydick = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')
print(mobydick)

#Creating training data and vocabulary
train, vocab = padded_everygram_pipeline(2, mobydick)

#Creating an empty vocabulary
lm = MLE(2)

#fitting the model + filling the vocabulary with the vocabulary from moby dick
lm.fit(train, vocab)

print(lm.vocab)

#printing length to see how many words it is containing
len(lm.vocab)

lm.vocab.lookup(mobydick[0])

lm.vocab.lookup(["aliens", "from", "Mars"])

#When it comes to ngram models the training boils down to counting up the ngrams from the training corpus.
print(lm.counts)

#This provides a convenient interface to access counts for unigrams…
lm.counts['a']

#…and bigrams (in this case “a b”)
lm.counts[['moby']]['dick']


