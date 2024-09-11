import json
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')  # Download 'punkt' for tokenization
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import random
import sys
import io


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
with open('data.json') as file:
    data = json.load(file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process each intent
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        # Add tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Perform lemmatization and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"Classes: {classes}")
print(f"Words: {words}")

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Initialize bag of words
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1 if word in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the data and convert to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)
X_train = np.array(list(training[:, 0]), dtype='float32')
y_train = np.array(list(training[:, 1]), dtype='float32')

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Redirect stdout to a file
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Save the data for future use
import pickle
pickle.dump({'words': words, 'classes': classes}, open('data.pkl', 'wb'))

print("Training complete and model saved!")


# Function to preprocess user input
def preprocess_input(user_input):
    # Tokenize and lemmatize the user input
    word_list = nltk.word_tokenize(user_input)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    
    # Create a bag of words
    bag = [1 if word in word_list else 0 for word in words]
    return np.array(bag).reshape(1, -1)

# Function to get response based on user input
def get_response(user_input):
    # Preprocess user input
    bag_of_words = preprocess_input(user_input)
    
    # Predict the class
    prediction = model.predict(bag_of_words)
    class_id = np.argmax(prediction)
    response_tag = classes[class_id]
    
    # Find response
    for intent in data['intents']:
        if intent['tag'] == response_tag:
            response = random.choice(intent['responses'])
            return response

# Main loop for user interaction
def main():
    print("Chatbot is running. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()



