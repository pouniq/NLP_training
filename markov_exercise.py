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



def make_pi(encoded_lines,y,poet_id=0):
    first_word = {}
    
    for words, poet in zip(encoded_lines,y ):
        if poet_id != poet:
            continue
        if words:
            
            word = words[0]
                
            if word in first_word:
                first_word[word] += 1
            else:
                first_word[word] = 1
                
    total = sum(first_word.values())
    
    
    proba = {
        word: count / total
        for word, count in first_word.items()
    }
    
   
    return proba

make_pi(encoded_lines_train, y_train, poet_id=0)
make_pi(encoded_lines_test, y_test, poet_id=0)




def make_A(encoded_lines, y, poet_id):
    transitions = {}
    
    for words, poet in zip(encoded_lines,y):
        if poet != poet_id:
            continue
        
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i+1]
            
            
            if current_word not in transitions:
                transitions[current_word] = {}
                
            if next_word not in transitions[current_word]:
                transitions[current_word][next_word] = 1
            else:
                transitions[current_word][next_word] += 1
        # Convert counts to probabilities
    A = {}

    for current_word, next_words in transitions.items():

        total = sum(next_words.values())

        A[current_word] = {
            next_word: count / total
            for next_word, count in next_words.items()
        }

    return A
        

make_A(encoded_lines_train, y_train, poet_id=0)
make_A(encoded_lines_test, y_test, poet_id=0) 






