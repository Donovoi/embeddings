import tkinter as tk
from tkinter import filedialog
from sentence_transformers import SentenceTransformer
import cudf
import cupy as cp
from PIL import Image
import json
import cuml
import numpy as np



def choose_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return file_paths


selected_files = choose_files()
print("Selected files:", selected_files)

# Load the model
model = SentenceTransformer('all-mpnet-base-v2')


def load_file_content(file):
    if file.endswith('.csv'):
        df = cudf.read_csv(file)
        return df.to_pandas().to_string(index=False)
    elif file.endswith('.json'):
        with open(file, 'r') as f:
            content = json.load(f)
        return json.dumps(content)
    elif file.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(file)
        img_array = np.array(img)
        img_flat = img_array.flatten().astype(str)
        return " ".join(img_flat)
    else:
        with open(file, 'r') as f:
            return f.read()


def embed_files(files):
    embeddings = cudf.DataFrame()
    for file in files:
        content = load_file_content(file)
        embedding = model.encode(content)
        embedding_gpu = cp.asarray(embedding)  # Convert to GPU array
        embeddings = cudf.concat(
            [embeddings, cudf.DataFrame(embedding_gpu.get(), index=[file])])
    return embeddings


file_embeddings = embed_files(selected_files)
print("File embeddings generated.")


def query_files(query, file_embeddings):
    # Embed the query
    query_embedding = cp.asarray(model.encode(query))

    # Use cuML NearestNeighbors for similarity search
    nn = cuml.NearestNeighbors(n_neighbors=1, metric='cosine')
    nn.fit(file_embeddings)

    distances, indices = nn.kneighbors(cp.array([query_embedding]))

    closest_file = file_embeddings.index[indices[0][0]]
    closest_distance = distances[0][0]

    return closest_file, closest_distance


# Example of querying
query = "example search text"
closest_file, closest_distance = query_files(query, file_embeddings)
print(f"Closest file: {closest_file} with distance: {closest_distance}")
