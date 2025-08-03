import pandas as pd
import torch
from torch import nn
import numpy as np
from pynndescent import NNDescent
import streamlit as st
import pickle
import os

class MusicEmbeddingModel(nn.Module):
    def __init__(self, num_numeric, genre_vocab, emotion_vocab, goodfor_vocab, emb_dim=32):
        super().__init__()
        self.genre_emb = nn.EmbeddingBag(genre_vocab, emb_dim, mode='mean')
        self.emotion_emb = nn.EmbeddingBag(emotion_vocab, emb_dim, mode='mean')
        self.goodfor_emb = nn.EmbeddingBag(goodfor_vocab, emb_dim, mode='mean')
        self.linear_numeric = nn.Linear(num_numeric, emb_dim)

    def forward(self, x_numeric, genre_idx, genre_off, emotion_idx, emotion_off, goodfor_idx, goodfor_off):
        genre_vec = self.genre_emb(genre_idx, genre_off)
        emotion_vec = self.emotion_emb(emotion_idx, emotion_off)
        goodfor_vec = self.goodfor_emb(goodfor_idx, goodfor_off)
        numeric_vec = self.linear_numeric(x_numeric)
        return torch.cat([numeric_vec, genre_vec, emotion_vec, goodfor_vec], dim=1)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('spotify_dataset.csv')
    except FileNotFoundError:
        st.error("Please upload 'spotify_dataset.csv' to the same directory as this app.")
        return None, None, None, None
    
    # Data preprocessing (simplified from notebook)
    df = df.drop(columns=['text', 'Key', 'Time signature', 'Explicit'], errors='ignore')
    
    # Rename columns
    column_mapping = {
        'Artist(s)': 'artist', 'song': 'song_title', 'Length': 'length',
        'Genre': 'genre', 'Album': 'album', 'Release Date': 'release_date',
        'Tempo': 'tempo', 'Loudness (db)': 'loudness', 'Popularity': 'popularity',
        'Energy': 'energy', 'Danceability': 'danceability', 'Positiveness': 'positiveness',
        'Speechiness': 'speechiness', 'Liveness': 'liveness', 'Acousticness': 'acousticness',
        'Instrumentalness': 'instrumentalness', 'Good for Party': 'party',
        'Good for Work/Study': 'work_study', 'Good for Relaxation/Meditation': 'relaxation_meditation',
        'Good for Exercise': 'exercise', 'Good for Running': 'running',
        'Good for Yoga/Stretching': 'yoga_stretching', 'Good for Driving': 'driving',
        'Good for Social Gatherings': 'social_gatherings', 'Good for Morning Routine': 'morning_routine'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Process genres (simplified)
    if 'genre' in df.columns:
        df['genre_list'] = df['genre'].str.split(',')
        df_exploded = df.explode('genre_list')
        df_exploded['genre_list'] = df_exploded['genre_list'].str.strip()
        genre_dummies = pd.get_dummies(df_exploded['genre_list'], prefix='genre')
        genre_encoded = genre_dummies.groupby(df_exploded.index).sum()
        df = df.drop(columns=['genre', 'genre_list']).join(genre_encoded)
    
    # Process emotions
    if 'emotion' in df.columns:
        emotion_dummies = pd.get_dummies(df['emotion'], prefix='emotion', dtype=int)
        df = pd.concat([df.drop(columns=['emotion']), emotion_dummies], axis=1)
    
    # Clean numeric data
    if 'loudness' in df.columns:
        df["loudness"] = df["loudness"].astype(str).str.replace("db", "", case=False).str.strip()
        df["loudness"] = pd.to_numeric(df["loudness"], errors='coerce')
    
    # Fill missing values
    numeric_cols = ['tempo', 'loudness', 'popularity', 'energy', 'danceability',
                    'positiveness', 'speechiness', 'liveness', 'acousticness', 'instrumentalness']
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    # Get column groups
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    emotion_cols = [col for col in df.columns if col.startswith("emotion_")]
    goodfor_cols = ['party', 'work_study', 'relaxation_meditation', 'exercise', 'running', 
                    'yoga_stretching', 'driving', 'social_gatherings', 'morning_routine']
    goodfor_cols = [col for col in goodfor_cols if col in df.columns]
    
    return df, numeric_cols, genre_cols, emotion_cols, goodfor_cols

@st.cache_resource
def build_recommendation_system(df, numeric_cols, genre_cols, emotion_cols, goodfor_cols):
    """Build the recommendation system"""
    device = torch.device("cpu")  # Use CPU for simplicity
    
    # Prepare data
    def prepare_embeddingbag_inputs_vectorized(df, col_group):
        data = df[col_group].values.astype(np.int64)
        row_ids, col_ids = np.nonzero(data)
        indices_tensor = torch.tensor(col_ids, dtype=torch.long)
        offsets = torch.zeros(len(df), dtype=torch.long)
        np.add.at(offsets.numpy(), row_ids, 1)
        offsets = torch.cumsum(offsets, dim=0) - offsets
        return indices_tensor.to(device), offsets.to(device)
    
    X_numeric = torch.tensor(df[numeric_cols].astype(float).values, dtype=torch.float32).to(device)
    genre_indices, genre_offsets = prepare_embeddingbag_inputs_vectorized(df, genre_cols)
    emotion_indices, emotion_offsets = prepare_embeddingbag_inputs_vectorized(df, emotion_cols)
    goodfor_indices, goodfor_offsets = prepare_embeddingbag_inputs_vectorized(df, goodfor_cols)
    
    # Create model
    model = MusicEmbeddingModel(
        num_numeric=len(numeric_cols),
        genre_vocab=len(genre_cols),
        emotion_vocab=len(emotion_cols),
        goodfor_vocab=len(goodfor_cols),
        emb_dim=32
    ).to(device)
    
    # Generate embeddings
    def batch_embedding_inputs(indices, offsets, batch_start, batch_end):
        batch_offsets = offsets[batch_start:batch_end]
        next_offset = offsets[batch_end] if batch_end < len(offsets) else len(indices)
        batch_indices = indices[batch_offsets[0]:next_offset]
        batch_offsets = batch_offsets - batch_offsets[0]
        return batch_indices, batch_offsets
    
    batch_size = 1024
    embedding_batches = []
    
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        xb = X_numeric[i:batch_end]
        
        genre_idx_batch, genre_off_batch = batch_embedding_inputs(genre_indices, genre_offsets, i, batch_end)
        emotion_idx_batch, emotion_off_batch = batch_embedding_inputs(emotion_indices, emotion_offsets, i, batch_end)
        goodfor_idx_batch, goodfor_off_batch = batch_embedding_inputs(goodfor_indices, goodfor_offsets, i, batch_end)
        
        with torch.no_grad():
            emb = model(xb, genre_idx_batch, genre_off_batch, emotion_idx_batch, emotion_off_batch,
                       goodfor_idx_batch, goodfor_off_batch)
            embedding_batches.append(emb.cpu())
    
    final_embeddings = torch.cat(embedding_batches)
    
    # Build index
    data = final_embeddings.cpu().numpy()
    index = NNDescent(data, metric='cosine', n_neighbors=10)
    
    return index, data

def get_recommendations(index, data, df, song_idx, k=5):
    """Get song recommendations"""
    indices, distances = index.query(data[song_idx:song_idx+1], k=k+1)  # +1 to exclude self
    
    recommendations = []
    for i, dist in zip(indices.flatten()[1:], distances.flatten()[1:]):  # Skip first (self)
        song_info = {
            'artist': df.iloc[i]['artist'],
            'song_title': df.iloc[i]['song_title'],
            'album': df.iloc[i]['album'],
            'similarity': 1 - dist  # Convert distance to similarity
        }
        recommendations.append(song_info)
    
    return recommendations

def main():
    st.title("🎵 Simple Music Recommendation App")
    st.write("Find similar songs based on audio features, genres, and emotions!")
    
    # Load data
    with st.spinner("Loading data..."):
        df, numeric_cols, genre_cols, emotion_cols, goodfor_cols = load_data()
    
    if df is None:
        return
    
    st.success(f"Loaded {len(df)} songs!")
    
    # Build recommendation system
    with st.spinner("Building recommendation system..."):
        index, data = build_recommendation_system(df, numeric_cols, genre_cols, emotion_cols, goodfor_cols)
    
    st.success("Recommendation system ready!")
    
    # Song selection
    st.header("Select a Song")
    
    # Create searchable dropdown
    df['display_name'] = df['artist'] + " - " + df['song_title']
    song_options = df['display_name'].tolist()
    
    selected_song = st.selectbox("Choose a song:", song_options)
    
    if selected_song:
        song_idx = df[df['display_name'] == selected_song].index[0]
        
        # Display selected song info
        st.subheader("Selected Song:")
        selected_info = df.iloc[song_idx]
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Artist:** {selected_info['artist']}")
            st.write(f"**Song:** {selected_info['song_title']}")
            st.write(f"**Album:** {selected_info['album']}")
        
        with col2:
            st.write(f"**Popularity:** {selected_info.get('popularity', 'N/A')}")
            st.write(f"**Energy:** {selected_info.get('energy', 'N/A')}")
            st.write(f"**Tempo:** {selected_info.get('tempo', 'N/A')}")
        
        # Get recommendations
        num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
        
        if st.button("Get Recommendations"):
            with st.spinner("Finding similar songs..."):
                recommendations = get_recommendations(index, data, df, song_idx, num_recommendations)
            
            st.subheader("Recommended Songs:")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['artist']} - {rec['song_title']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Artist:** {rec['artist']}")
                        st.write(f"**Album:** {rec['album']}")
                    with col2:
                        st.write(f"**Similarity:** {rec['similarity']:.3f}")
                        st.progress(rec['similarity'])

if __name__ == "__main__":
    main()