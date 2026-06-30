import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import gensim.downloader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

print("Loading word embedding model...")
model = gensim.downloader.load("glove-wiki-gigaword-50")
print("Model loaded successfully.")

words = ["king", "queen", "man", "woman", "emperor","prince","princess","boy","girl"]

# Collect embeddings for these words from the loaded gensim model
embeddings_list = []
valid_words = []
for w in words:
    if w in model: # Correctly check if word exists in the gensim model
        embeddings_list.append(model[w])
        valid_words.append(w)
    else:
        print(f"Warning: '{w}' not found in model vocabulary. Skipping this word.")

embeddings = np.array(embeddings_list)

# Check if there are enough samples (words) for 3D PCA
if len(valid_words) < 3:
    print(f"Not enough valid words ({len(valid_words)}) for 3D PCA. Need at least 3 to perform a 3D reduction and plot.")
    # If there aren't enough words, we can't create a meaningful 3D plot.
    # You might consider plotting in 2D if len(valid_words) >= 2, or displaying a message.
else:
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)

    fig = go.Figure(data=[go.Scatter3d(
        x=reduced[:,0], y=reduced[:,1], z=reduced[:,2],
        mode='markers+text',
        text=valid_words, # Use valid_words for annotations
        textposition='top center',
        marker=dict(
            size=8,
            color=reduced[:,2],       # color by one axis for depth cues
            colorscale='Viridis',
            opacity=0.9
        )
    )])

    fig.update_layout(
        title='3D PCA of Word Embeddings',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
