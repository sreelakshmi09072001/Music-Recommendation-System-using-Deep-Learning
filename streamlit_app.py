import streamlit as st
import pandas as pd
import torch
from torch import nn
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple

# Set page config
st.set_page_config(
    page_title="🎵 Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_model_and_data():
    """Load all necessary components for the recommendation system."""
    try:
        # Load model
        checkpoint = torch.load('music_embedding_model.pth', map_location='cpu')
        model_config = checkpoint['model_config']
        
        model = MusicEmbeddingModel(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load other components
        with open('nn_index.pkl', 'rb') as f:
            nn_index = pickle.load(f)
        
        final_embeddings = torch.load('final_embeddings.pth', map_location='cpu')
        
        with open('preprocessing_data.json', 'r') as f:
            preprocessing_data = json.load(f)
        
        songs_df = pd.read_pickle('songs_data.pkl')
        
        return model, nn_index, final_embeddings, preprocessing_data, songs_df
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure all model files are in the same directory.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

def search_songs(songs_df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search for songs by artist or title."""
    if not query:
        return songs_df.head(50)  # Return first 50 songs if no query
    
    query = query.lower()
    mask = (
        songs_df['artist'].str.lower().str.contains(query, na=False) |
        songs_df['song_title'].str.lower().str.contains(query, na=False) |
        songs_df['album'].str.lower().str.contains(query, na=False)
    )
    return songs_df[mask]

def get_recommendations(song_index: int, nn_index, final_embeddings, songs_df: pd.DataFrame, k: int = 5) -> List[Dict]:
    """Get recommendations for a given song index."""
    try:
        data = final_embeddings.cpu().numpy()
        indices, distances = nn_index.query(data[song_index:song_index+1], k=k+1)  # +1 to exclude the song itself
        
        recommendations = []
        for i, (idx, dist) in enumerate(zip(indices.flatten()[1:], distances.flatten()[1:])):  # Skip first (itself)
            song_info = songs_df.iloc[idx]
            recommendations.append({
                'rank': i + 1,
                'artist': song_info['artist'],
                'song_title': song_info['song_title'],
                'album': song_info['album'],
                'length': song_info['length'],
                'release_date': song_info['release_date'],
                'similarity_score': 1 - dist,  # Convert distance to similarity
                'tempo': song_info.get('tempo', 'N/A'),
                'energy': song_info.get('energy', 'N/A'),
                'danceability': song_info.get('danceability', 'N/A')
            })
        
        return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

def main():
    st.title("🎵 Music Recommendation System")
    st.markdown("Find similar songs based on audio features, genres, emotions, and listening contexts!")
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, nn_index, final_embeddings, preprocessing_data, songs_df = load_model_and_data()
    
    if model is None:
        st.stop()
    
    st.success(f"✅ Loaded {len(songs_df):,} songs successfully!")
    
    # Sidebar filters
    st.sidebar.header("🔍 Search & Filter")
    
    # Search functionality
    search_query = st.sidebar.text_input(
        "Search by artist, song, or album:",
        placeholder="e.g., Taylor Swift, Bohemian Rhapsody"
    )
    
    # Filter songs based on search
    filtered_songs = search_songs(songs_df, search_query)
    
    if filtered_songs.empty:
        st.warning("No songs found matching your search. Please try a different query.")
        st.stop()
    
    st.sidebar.write(f"Found {len(filtered_songs):,} songs")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎯 Select a Song")
        
        # Song selection
        if len(filtered_songs) > 0:
            # Create a display format for songs
            song_options = []
            for idx, row in filtered_songs.iterrows():
                display_text = f"{row['artist']} - {row['song_title']}"
                if pd.notna(row['album']):
                    display_text += f" ({row['album']})"
                song_options.append((display_text, idx))
            
            selected_display = st.selectbox(
                "Choose a song:",
                options=[option[0] for option in song_options],
                help="Select a song to get recommendations"
            )
            
            # Get the actual index
            selected_idx = next(idx for display, idx in song_options if display == selected_display)
            selected_song = songs_df.iloc[selected_idx]
            
            # Display selected song info
            st.subheader("📀 Selected Song")
            st.write(f"**Artist:** {selected_song['artist']}")
            st.write(f"**Title:** {selected_song['song_title']}")
            st.write(f"**Album:** {selected_song['album']}")
            st.write(f"**Length:** {selected_song['length']}")
            st.write(f"**Release Date:** {selected_song['release_date']}")
            
            # Audio features if available
            if 'tempo' in selected_song and pd.notna(selected_song['tempo']):
                st.write(f"**Tempo:** {selected_song['tempo']} BPM")
            if 'energy' in selected_song and pd.notna(selected_song['energy']):
                st.write(f"**Energy:** {selected_song['energy']:.2f}")
            if 'danceability' in selected_song and pd.notna(selected_song['danceability']):
                st.write(f"**Danceability:** {selected_song['danceability']:.2f}")
    
    with col2:
        st.header("🎶 Recommendations")
        
        # Number of recommendations
        num_recommendations = st.slider("Number of recommendations:", 3, 20, 10)
        
        if st.button("🔮 Get Recommendations", type="primary"):
            with st.spinner("Finding similar songs..."):
                recommendations = get_recommendations(
                    selected_idx, nn_index, final_embeddings, songs_df, num_recommendations
                )
            
            if recommendations:
                st.success(f"Found {len(recommendations)} similar songs!")
                
                # Display recommendations
                for i, rec in enumerate(recommendations):
                    with st.expander(f"#{rec['rank']} - {rec['artist']} - {rec['song_title']}", expanded=i < 3):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Artist:** {rec['artist']}")
                            st.write(f"**Album:** {rec['album']}")
                            st.write(f"**Length:** {rec['length']}")
                            st.write(f"**Release Date:** {rec['release_date']}")
                        
                        with col_b:
                            st.write(f"**Similarity Score:** {rec['similarity_score']:.3f}")
                            if rec['tempo'] != 'N/A':
                                st.write(f"**Tempo:** {rec['tempo']} BPM")
                            if rec['energy'] != 'N/A':
                                st.write(f"**Energy:** {rec['energy']:.2f}")
                            if rec['danceability'] != 'N/A':
                                st.write(f"**Danceability:** {rec['danceability']:.2f}")
                
                # Create a summary table
                st.subheader("📊 Summary Table")
                rec_df = pd.DataFrame(recommendations)
                display_cols = ['rank', 'artist', 'song_title', 'album', 'similarity_score']
                st.dataframe(
                    rec_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error("Could not generate recommendations. Please try another song.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🎵 Powered by PyTorch Neural Embeddings & Nearest Neighbor Search</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()