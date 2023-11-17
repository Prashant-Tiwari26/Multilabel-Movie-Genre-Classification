import numpy as np

def get_word_to_index(glove_path:str):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    word_to_index = {word: i for i, word in enumerate(embeddings_index.keys())}
    return word_to_index

def pad_sequence(sequence, max_seq_length):
    if len(sequence) > max_seq_length:
        return sequence[:max_seq_length]
    else:
        padding = max_seq_length - len(sequence)
        return sequence + ['<PAD>'] * padding