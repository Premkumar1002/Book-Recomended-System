import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Ignore warnings
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv('ML_Project/ratings.csv')

# Display first few rows of the dataset
print(dataset.head())

# Get the shape of the dataset
print("Dataset shape:", dataset.shape)

# Split the dataset into training and testing sets
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Display first few rows of train and test datasets
print(train.head())
print(test.head())

# Get the number of unique users and books
n_users = len(dataset.user_id.unique())
n_books = len(dataset.book_id.unique())

# Print number of users and books
print("Number of users:", n_users)
print("Number of books:", n_books)

# Create book embedding
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books + 1, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

# Create user embedding
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users + 1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# Perform dot product between user and book embeddings
prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])

# Build and compile the model
model = Model([user_input, book_input], prod)
model.compile(optimizer='adam', loss='mean_squared_error')

# Load or train the model
if os.path.exists('regression_model.h5'):
    model = load_model('regression_model.h5')
else:
    history = model.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)
    model.save('regression_model.h5')
    # Plot training loss
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")
    plt.show()

# Evaluate the model
model.evaluate([test.user_id, test.book_id], test.rating)

# Make predictions for the first 10 test samples
predictions = model.predict([test.user_id.head(10), test.book_id.head(10)])

# Print predictions alongside actual ratings
[print(predictions[i], test.rating.iloc[i]) for i in range(10)]

# Second model with additional layers
conc = Concatenate()([book_vec, user_vec])
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

model2 = Model([user_input, book_input], out)
model2.compile(optimizer='adam', loss='mean_squared_error')

# Load or train the second model
if os.path.exists('regression_model2.h5'):
    model2 = load_model('regression_model2.h5')
else:
    history = model2.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)
    model2.save('regression_model2.h5')
    # Plot training loss
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")
    plt.show()

# Evaluate the second model
model2.evaluate([test.user_id, test.book_id], test.rating)

# Make predictions with the second model
predictions2 = model2.predict([test.user_id.head(10), test.book_id.head(10)])
[print(predictions2[i], test.rating.iloc[i]) for i in range(10)]

# Extract book embeddings
book_em = model.get_layer('Book-Embedding')
book_em_weights = book_em.get_weights()[0]

# Apply PCA to reduce dimensionality
pca = PCA(n_components=2)
pca_result = pca.fit_transform(book_em_weights)

# Scatter plot for PCA results
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
plt.show()

# Normalize book embeddings
book_em_weights = book_em_weights / np.linalg.norm(book_em_weights, axis=1).reshape((-1, 1))

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tnse_results = tsne.fit_transform(book_em_weights)

# Scatter plot for t-SNE results
sns.scatterplot(x=tnse_results[:, 0], y=tnse_results[:, 1])
plt.show()

# Create dataset for recommendations
book_data = np.array(list(set(dataset.book_id)))
user = np.array([1 for _ in range(len(book_data))])

# Generate predictions for user 1
predictions = model.predict([user, book_data])

# Get top 5 recommended book IDs
predictions = np.array([a[0] for a in predictions])
recommended_book_ids = (-predictions).argsort()[:5]

# Display recommended book IDs and their scores
print("Recommended book IDs:", recommended_book_ids)
print("Predicted scores:", predictions[recommended_book_ids])

# Load book details
books = pd.read_csv('ML_Project/books.csv')

# Display recommended books
print(books[books['id'].isin(recommended_book_ids)])
