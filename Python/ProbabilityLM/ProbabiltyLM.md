# Probabilty Language Models
Probability Language Models are a type of language model that uses probabilities to predict the next word in a sequence by calculating the 
mathematical probabilty of the next word given $n$ previous words. Thus they are called n-gram models.

## Mathematical Background
Denote $P(w)$ as the probability of the next word in a sequence being $w$ given the previous $n$ words.
In order to calculate the probaility of this we need to count the number of times $w$ occurs in the training data in the context of the previous $n$ words.
We choose the most probable word as the next word and repeat this process.

### Algebraic Representation
there is not much to be said here. We simply denote by $C(w_1, ..., w_n)$ the number of times $w_1, ..., w_n$ occurs in the training data and by $w_curr$ the current word. 
Then we can calculate the probability of $w$ given the previous $n$ words as follows:

$$
P(w) = \frac{C(w_{curr-n}, ..., w_{curr}, w)}{C(w_{curr-n}, ..., w_{curr})}
$$

since the every possible outcome is represented in the denominator adn we want to know the probability of the outcome of $w$ given the previous $n$ words.


