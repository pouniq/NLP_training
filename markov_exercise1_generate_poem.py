import string
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


poems = [
    'robert_frost.txt',
    'edgar_allan_poe.txt'
]


def preprocess_text(file_add):
    
    

    with open(file_add) as f:
     text0 =  f.read()
    
    text0 = text0.lower()
    
    
    text0 = text0.translate(
    str.maketrans('','',string.punctuation)
    )
    
    lines = text0.split('\n')

    # remove empty lines and extra whitespace
    lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    return lines
    
    
text0 = preprocess_text(poems[0])
text1 = preprocess_text(poems[1])



def first_word_prob(text):
    
    first_word_dict = {}
    
    for line in text:
        first_word = line.split()[0]
        
        if first_word in first_word_dict:
            first_word_dict[first_word] += 1
        else:
            first_word_dict[first_word] = 1
    
    total = sum(first_word_dict.values())
    
    
    first_word_prob = {}
    
    for word, counts in first_word_dict.items():
        first_word_prob[word] = counts / total
        
    return first_word_prob
    

p_text_0 = first_word_prob(text0)
p_text_1 = first_word_prob(text1)



def make_A(text):
    
    A = {}
    
    for line in text:
        words = line.split()
        
        for i in range(len(words)-2):
            
            word1 = words[i]
            word2 = words[i+1]
            word3 = words[i+2]
        
        
            if word1 not in A:
                A[word1] = {}
            
            if word2 not in A[word1]:
                A[word1][word2] = {}
            
            if word3 not in A[word1][word2]:
                A[word1][word2][word3] = 1
            else:
                A[word1][word2][word3] += 1
        
    return A
            

A_text_0 = make_A(text0)
A_text_1 = make_A(text1)



def prob_A(text):
    
    for word1 in text:
        for word2 in text[word1]:
            
            total = sum(text[word1][word2].values())
            
            for word3 in text[word1][word2]:
                
                text[word1][word2][word3] /= total
            
    return text
    

prob_A_text_0 = prob_A(A_text_0)
prob_A_text_1 = prob_A(A_text_1)


def sample_word(prob_dict):
    
    p_sample = np.random.random()
    cumulative = 0
    for word, probability in prob_dict.items():
        cumulative += probability
        if p_sample < cumulative:
            return word
    # floating point edge case: fall back to last word
    return word


def make_A(text):
    A = {}
    for line in text:
        words = line.split()
        for i in range(len(words) - 2):          # <-- now runs the body every iteration
            word1, word2, word3 = words[i], words[i+1], words[i+2]

            A.setdefault(word1, {})
            A[word1].setdefault(word2, {})
            A[word1][word2][word3] = A[word1][word2].get(word3, 0) + 1
    return A

def second_word_prob(text):
    """P(second word | first word) — needed to seed line generation."""
    second_word_dict = {}
    for line in text:
        words = line.split()
        if len(words) < 2:
            continue
        w1, w2 = words[0], words[1]
        second_word_dict.setdefault(w1, {})
        second_word_dict[w1][w2] = second_word_dict[w1].get(w2, 0) + 1

    for w1, counts in second_word_dict.items():
        total = sum(counts.values())
        for w2 in counts:
            counts[w2] /= total
    return second_word_dict

p2_text_0 = second_word_prob(text0)
p2_text_1 = second_word_prob(text1)

def generate_line(first_word_probs, second_word_probs, transitions, max_len=15):
    word0 = sample_word(first_word_probs)
    if word0 not in second_word_probs:
        return word0
    word1 = sample_word(second_word_probs[word0])
    words = [word0, word1]

    while len(words) < max_len:
        if word0 not in transitions or word1 not in transitions[word0]:
            break
        word2 = sample_word(transitions[word0][word1])
        words.append(word2)
        word0, word1 = word1, word2

    return ' '.join(words)

# usage
print(generate_line(p_text_0, p2_text_0, prob_A_text_0))