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
#4569

#…and bigrams (in this case “Moby Dick”)
lm.counts[['Moby']]['Dick']
#83

#However, the real purpose of training a language model is to have it score how probable words are in certain contexts. This being MLE, the model returns the item’s relative frequency as its score.
lm.score("whale")
#0.0032244632122913975

#Items that are not seen during training are mapped to the vocabulary’s “unknown label” token. This is “<UNK>” by default.
lm.score("<UNK>") == lm.score("Mars")
#True

#Here’s how you get the score for a word given some preceding context. For example we want to know what is the chance that “Moby” is preceded by “Dick”.
lm.score("Dick", ["Moby"])
#0.988



