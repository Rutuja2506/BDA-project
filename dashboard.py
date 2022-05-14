from email.mime import image
from tkinter import Image
from turtle import width
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# For base64 encoding
import base64

# Supress warnings
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set app title
# st.set_page_config(page_title='Spotify User Dashboard')

# Page background
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
        .stApp {
          background-image: url("data:image/png;base64,%s");
          background-size: cover;
        }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

st.markdown("<h1 style='text-align: center; color: white;'>Spotify EDA and Recommendation System</h1>", unsafe_allow_html=True)
image=Image.open('img2.png')
# st.image(image,width=100)


col1, col2, col3 = st.columns(3)

with col1:
    st.write('      ')

with col2:
    st.image(image,width=150)

    # st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.write(' ')
# st.title('Spotify User Dashboard')
#Spotify EDA for 600k+tracks
st.subheader('Spotify EDA for 600k+ Tracks from 1921-2020')

path2='tracks.csv'
df=pd.read_csv(path2)
df.duplicated().sum()
#Convert Milli secs duration into minutes
df['duration_min'] = df['duration_ms']/60000
df['duration_min'] = df['duration_min'].round(2)
# df_by_genres['duration_min'] = df_by_genres['duration_ms']/60000
# df_by_genres['duration_min'] = df_by_genres['duration_min'].round(2)
#Remove the Square Brackets from the artists

df["artists"]=df["artists"].str.replace("[", "")
df["artists"]=df["artists"].str.replace("]", "")
df["artists"]=df["artists"].str.replace("'", "")
#Drop the columns
df.drop(['duration_ms'],inplace=True,axis=1)
df['date'] = pd.to_datetime(df['release_date'],format='%Y-%m-%d')
df['year'] = pd.DatetimeIndex(df['date']).year

col1, col2 = st.columns(2)

#Most Popular Tracks
# st.title('Top 10 Popular Tracks')
col1.subheader("Top 10 Popular Tracks")
popular = df.groupby("name")['popularity'].mean().sort_values(ascending=False).head(10)

fig=px.bar(popular,labels={"name":"Tracks","value":"Popularity"},title='Top 10 Popular Tracks',color=popular.index,width=700, height=700)
fig.update_xaxes(tickangle=-90)
# fig.update_layout(
#         margin=dict(l=20, r=20, t=20, b=20))
# fig.update(layout_showlegend=False)
fig.update_layout(showlegend=False, paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0)' )
# fig.show()
col1.write(fig)

# #Most frequent 10 artists
col2.subheader('Most frequent 10 artists')
artist_max_songs=pd.DataFrame(df['artists'].value_counts().head(10)).reset_index()
artist_max_songs.columns=['Artists','Songs_Count']
fig = px.pie(artist_max_songs,values='Songs_Count', names='Artists')
fig.update_layout(showlegend=False, paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0)' )
# fig.show()
col2.write(fig)




# visualize the popularity of The Beatles songs over the year
# Beatles = df[df['artists'] == 'The Beatles']
die_drei = df[df['artists'] == 'Die drei???']
plt.rcParams['figure.figsize'] = (11,7)
# line plot passing x,y
sns.lineplot(x='year', y='popularity', data=die_drei, color='green')
# Labels
plt.title("The Die Drei?? Popularity")
plt.xlabel('Year')
plt.ylabel('Popularity')
plt.xticks(rotation = 45)
plt.show()


#Artists with maximum number of songs
st.header('Artists with maximum number of songs')
artist_max_songs=pd.DataFrame(df['artists'].value_counts().head()).reset_index()
artist_max_songs.columns=['Artists','Songs Count']
st.dataframe(artist_max_songs)

#Most Popular Artists
# st.title('Most Popular Artists')
st.subheader('Most Popular Artists')
popular = df.groupby("artists")['popularity'].sum().sort_values(ascending=False)[:20]

fig=px.bar(popular,labels={"name":"Tracks","value":"Popularity"},title='Top 20 Artists with Popularity',color=popular.index,width=800, height=700)
fig.update_xaxes(tickangle=-90)
fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40))
fig.update(layout_showlegend=False)
# fig.update_layout(showlegend=False, paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0)' )
st.write(fig)

col3, col4 = st.columns(2)

#Number of songs based on Year of Release Date
col3.subheader('Number of songs based on Year of Release Date')
year = pd.DataFrame(df['year'].value_counts())
year = year.sort_index()
fig=px.line(year,labels={"index":"Years","value":"Count"},title='Number of songs released Yearwise',width=600, height=500)
# fig.update_xaxes(tickangle=-90)
fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40))
col3.write(fig)

#Finding Top 10 genres
col4.header('Top 10 genres')
df_genres=pd.read_csv('data_by_genres.csv')
top10_genres = df_genres.nlargest(10, 'popularity')
fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.update_layout(showlegend=False, paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0)' )
col4.write(fig)


#Exploring Data by year dataset

path3='data_by_year.csv'
df_year=pd.read_csv(path3)
df_year_control = df_year.copy()
df_year_control['acousticness'] = df_year_control['acousticness'] / df_year_control['acousticness'].max()
df_year_control['danceability'] = df_year_control['danceability'] / df_year_control['danceability'].max()
df_year_control['duration_ms'] = df_year_control['duration_ms'] / df_year_control['duration_ms'].max()
df_year_control['energy'] = df_year_control['energy'] / df_year_control['energy'].max()
df_year_control['instrumentalness'] = df_year_control['instrumentalness'] / df_year_control['instrumentalness'].max()
df_year_control['liveness'] = df_year_control['liveness'] / df_year_control['liveness'].max()
df_year_control['speechiness'] = df_year_control['speechiness'] / df_year_control['speechiness'].max()
df_year_control['tempo'] = df_year_control['tempo'] / df_year_control['tempo'].max()
df_year_control['valence'] = df_year_control['valence'] / df_year_control['valence'].max()
df_year_control['popularity'] = df_year_control['popularity'] / df_year_control['popularity'].max()
df_year_control['loudness'] = df_year_control['loudness'] / df_year_control['loudness'].min()
df_year_control['year'] = df_year_control['year'].astype(str)

df_year_control.drop(["key","mode"],axis=1,inplace=True)
df_year_control = df_year_control.melt("year")
st.subheader('Comparing Characteristics every 10 years')
fig = px.line_polar(df_year_control, r="value",theta="variable",line_close=True,
             animation_frame="year",template="plotly_dark",range_r = (0,1))
fig.update_traces(fill='toself')
fig.update_layout(font_size=15)
# fig.show()
st.write(fig)






#Clustering 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = df_genres.select_dtypes(np.number)
cluster_pipeline.fit(X)
df_genres['cluster'] = cluster_pipeline.predict(X)
# Visualizing the Clusters with t-SNE
# st.header('Clustering Genres Using K-means')
from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = df_genres['genres']
projection['cluster'] = df_genres['cluster']

fig_gen = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'],title='Clustering Genres Using K-means')

# st.write(fig)

#Kmeans clustering for Songs
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)

X = df.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
df['cluster_label'] = song_cluster_labels

# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = df['name']
projection['cluster'] = df['cluster_label']
# st.header('Clustering Songs Using K-means')
fig_songs = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'],title='Clustering Songs Using K-means')

# st.write(fig)

#Radio button implementation for selecting two Clustering
graph_selection = st.radio(
     "Select your Clusters",
    ('Clustering for Genres', 'Clustering for Songs'))

if graph_selection == 'Clustering for Genres':
    st.plotly_chart(fig_gen)
else:
    st.plotly_chart(fig_songs)


#Building Recommenddation System
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# from collections import defaultdict
# import os
# import spotipy.util
# import seaborn as sns
# sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="be66f19920224165a8b3610551084bf4",
#                                                            client_secret="f4af029888934b86b63bec11b21445aa"))

# def find_song(name, year):
#     song_data = defaultdict()
#     results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
# #     print(results)
#     if results['tracks']['items'] == []:
#         return None

#     results = results['tracks']['items'][0]
#     track_id = results['id']
#     audio_features = sp.audio_features(track_id)[0]

#     song_data['name'] = [name]
#     song_data['year'] = [year]
#     song_data['explicit'] = [int(results['explicit'])]
# #     song_data['duration_min'] = [results['duration_min']]
#     song_data['popularity'] = [results['popularity']]

#     for key, value in audio_features.items():
#         song_data[key] = value

#     return pd.DataFrame(song_data)





























st.markdown(
    '''
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
<div class='jumbotron text-center footer' style='background-color: ##000000;'>
    <div class='row'>
        <div class='col-md-12'>
            <p style='font-weight: 400'><center>___________________</center></p>
            <p style='font-weight: 400'><center>Designed, Developed and Maintained by Rutuja Medhekar</center></p>
        </div>
    </div>
<div>
    ''',
    unsafe_allow_html=True
)
