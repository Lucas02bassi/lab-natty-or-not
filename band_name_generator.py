import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Dados de exemplo de nomes de bandas
band_names = [
    "The Rolling Stones",
    "Led Zeppelin",
    "Pink Floyd",
    "The Beatles",
    "Queen",
    "Metallica",
    "AC/DC",
    "Nirvana",
    "Guns N' Roses",
    "The Who",
    "Black Sabbath",
    "The Doors",
    "U2",
    "The Eagles",
    "The Beach Boys",
    "Fleetwood Mac",
    "Red Hot Chili Peppers",
    "Radiohead",
    "Foo Fighters",
    "The Police",
    "Deep Purple",
    "R.E.M.",
    "Pearl Jam",
    "The Clash",
    "Bon Jovi",
    "Genesis",
    "Van Halen",
    "The Rolling Stones",
    "Aerosmith",
    "The Cure",
    "Oasis",
    "Green Day",
    "Def Leppard",
    "The Velvet Underground",
    "The Smashing Pumpkins",
    "Journey",
    "Cream",
    "The Allman Brothers Band",
    "Blur",
    "The Smiths",
    "Ramones",
    "The Kinks",
    "The Strokes",
    "Nine Inch Nails",
    "The White Stripes",
    "Talking Heads",
    "Rush",
    "Queen",
    "Metallica"
]

# Pré-processamento dos dados
# Convertendo os nomes das bandas em sequências de caracteres
all_chars = sorted(set(''.join(band_names)))
char_to_index = {char: i for i, char in enumerate(all_chars)}
index_to_char = {i: char for char, i in char_to_index.items()}
max_sequence_length = max([len(name) for name in band_names])

# Convertendo nomes de bandas em sequências de índices de caracteres
X = np.zeros((len(band_names), max_sequence_length, len(all_chars)), dtype=bool)
Y = np.zeros((len(band_names), max_sequence_length, len(all_chars)), dtype=bool)
for i, name in enumerate(band_names):
    for t, char in enumerate(name):
        X[i, t, char_to_index[char]] = 1
        if t < len(name) - 1:
            Y[i, t, char_to_index[name[t + 1]]] = 1

# Construindo o modelo RNN
model = Sequential([
    Input(shape=(max_sequence_length, len(all_chars))),
    LSTM(128, return_sequences=True),
    Dense(len(all_chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Treinando o modelo
model.fit(X, Y, epochs=100, verbose=1)


# Função para gerar nomes de bandas
def generate_band_name(model, max_sequence_length, char_to_index, index_to_char):
    start_sequence = np.zeros((1, max_sequence_length, len(char_to_index)), dtype=bool)
    generated_name = ''
    for i in range(max_sequence_length):
        prediction = model.predict(start_sequence)
        next_char_index = np.random.choice(len(char_to_index), p=prediction[0, i])
        next_char = index_to_char[next_char_index]
        generated_name += next_char
        start_sequence[0, i, next_char_index] = 1
    return generated_name


# Gerando um nome de banda
generated_band_name = generate_band_name(model, max_sequence_length, char_to_index, index_to_char)
print("Nome da Banda Gerado:", generated_band_name)
