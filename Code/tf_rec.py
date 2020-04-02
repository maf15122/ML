#import libraries 
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model as keras_load_model
from keras import losses
from keras.callbacks import EarlyStopping

#import data 
columns_rating = ['user_id', 'movie_id', 'rating']
columns_movie = ['movie_id', 'title']
ratings = pd.read_csv('u.data', sep='\t', names=columns_rating, usecols=range(3), encoding="ISO-8859-1")
movies = pd.read_csv('u.item', sep='|', names=columns_movie, usecols=range(2), encoding="ISO-8859-1")


rating_counts = ratings.groupby("movie_id")["rating"].count().sort_values(ascending=False)
pop_ratings = ratings[ratings["movie_id"].isin((rating_counts).index[0:100])]
pop_ratings = pop_ratings.set_index(["movie_id", "user_id"])

prefs = pop_ratings["rating"]
mean_0 = pop_ratings["rating"].mean()
prefs = prefs - mean_0
mean_i = prefs.groupby("movie_id").mean()
prefs = prefs - mean_i
mean_u = prefs.groupby("user_id").mean()
prefs = prefs - mean_u
pref_matrix = prefs.reset_index()[["user_id", "movie_id", "rating"]].pivot(index="user_id", columns="movie_id", values="rating")

ENCODING_DIM = 25
ITEM_COUNT = 100
input_layer = Input(shape=(ITEM_COUNT, ))
encoded = Dense(ENCODING_DIM, activation="linear", use_bias=False)(input_layer)
decoded = Dense(ITEM_COUNT, activation="linear", use_bias=False)(encoded)
recommender = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
encoded_input = Input(shape=(ENCODING_DIM, ))
decoder = Model(encoded_input, recommender.layers[-1](encoded_input))
prefs[prefs == 0]
