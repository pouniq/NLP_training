# ! wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/refs/heads/master/hmm_class/edgar_allan_poe.txt
# ! wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/refs/heads/master/hmm_class/robert_frost.txt
import string
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np



with open('edgar_allan_poe.txt') as f:
    text1 = f.read().lower()

with open('robert_frost.txt') as f:
    text2 = f.read().lower()
    
    
t1 = text1.split('\n')
t2 = text2.split('\n')







df = pd.DataFrame({
    'poetry_line': t1,
    'poet': 0
})

df_1 = pd.DataFrame({
    'poetry_line': t2,
    'poet': 1
})

df = pd.concat([df, df_1], ignore_index=True)



df.groupby('poet').count()
df[df['poetry_line'] == ''].sum()


X = df['poetry_line']
y = df['poet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)



X_train = X_train.apply(
    lambda line: line.translate(
        str.maketrans('', '', string.punctuation)
    )
)


X_test = X_test.apply(
    lambda line: line.translate(
        str.maketrans('', '', string.punctuation)
    )
)


## Train encoded

word_list_train = [word.rstrip().split() for word in X_train]

unique_words = set(
    word
    for line in word_list_train
    for word in line
)
    
word_2_int = {word:i
                for i,word in enumerate(unique_words)}

unk = len(word_2_int)
word_2_int['<unk>'] = unk

len(word_2_int)
encoded_lines_train = [
    [word_2_int[word] for word in line]
    for line in word_list_train
    if len(line) > 0
]


len(encoded_lines_train)
## test encoded

word_list_test = [word.rstrip().split() for word in X_test]

encoded_lines_test = [
    [word_2_int.get(word, unk) for word in line]
    for line in word_list_test
    if len(line) > 0
]


## Markov Model
V = len(word_2_int)


A_0 = np.ones((V,V))
P_0 = np.ones(V)




def make_pi(encoded_lines, y, poet_id, V):
    pi = np.zeros(V)

    for words, poet in zip(encoded_lines, y):
        if poet != poet_id:
            continue

        if words:
            first_word = words[0]
            pi[first_word] += 1

    total = pi.sum()

    if total > 0:
        pi /= total

    return pi

p0 = np.array(make_pi(encoded_lines_train, y_train, poet_id=0, V=V))
p1 = np.array(make_pi(encoded_lines_train, y_train, poet_id=1, V=V))




def make_A(encoded_lines, y, poet_id, V):
    A = np.ones((V, V))  # Laplace smoothing

    for words, poet in zip(encoded_lines, y):

        if poet != poet_id:
            continue

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]

            A[current_word, next_word] += 1

    # Normalize each row
    A /= A.sum(axis=1, keepdims=True)

    return A
        

A0 = make_A(encoded_lines_train, y_train, poet_id=0,V=V)
A1 = make_A(encoded_lines_train, y_train, poet_id=1,V=V)

A0 = np.log(A0)
A1 = np.log(A1)


## compute priors
prior0 = np.mean(y_train == 0)
prior1 = np.mean(y_train == 1)

logprior0 = np.log(prior0)
logprior1 = np.log(prior1)



def predict(X_test, word_2_int, pi0, pi1,
            A0, A1, prior0, prior1):

    predictions = []

    # Convert probabilities to log probabilities
    logpi0 = np.log(pi0)
    logpi1 = np.log(pi1)

    logA0 = np.log(A0)
    logA1 = np.log(A1)

    logprior0 = np.log(prior0)
    logprior1 = np.log(prior1)

    unk = word_2_int['<unk>']

    for line in X_test:

        # Split sentence into words
        words = line.split()

        # Convert words to integers
        encoded = [
            word_2_int.get(word, unk)
            for word in words
        ]

        # Empty sentence
        if len(encoded) == 0:
            predictions.append(0)
            continue

        # First word
        first_word = encoded[0]

        # -------------------------
        # Score for Poe
        # -------------------------

        score0 = logprior0 + logpi0[first_word]

        for i in range(len(encoded) - 1):

            current_word = encoded[i]
            next_word = encoded[i + 1]

            score0 += logA0[current_word, next_word]

        # -------------------------
        # Score for Frost
        # -------------------------

        score1 = logprior1 + logpi1[first_word]

        for i in range(len(encoded) - 1):

            current_word = encoded[i]
            next_word = encoded[i + 1]

            score1 += logA1[current_word, next_word]

        # -------------------------
        # Choose poet
        # -------------------------

        if score0 > score1:
            predictions.append(0)
        else:
            predictions.append(1)

    return np.array(predictions)


X_test = X_test.apply(
    lambda line: line.translate(
        str.maketrans('', '', string.punctuation)
    )
)


y_pred_train = predict(
    
    X_train,
    word_2_int,
    p0,
    p1,
    A0,
    A1,
    prior0,
    prior1
    
)

y_pred = predict(
    X_test,
    word_2_int,
    p0,
    p1,
    A0,
    A1,
    prior0,
    prior1
)

f1_score(y_train, y_pred_train)
f1_score(y_test, y_pred)