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
pop_ratings = ratings[ratings["movie_id"].isin((rating_counts).index[0:500])]
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
ITEM_COUNT = 500
input_layer = Input(shape=(ITEM_COUNT, ))
encoded = Dense(ENCODING_DIM, activation="linear", use_bias=False)(input_layer)
decoded = Dense(ITEM_COUNT, activation="linear", use_bias=False)(encoded)
recommender = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
encoded_input = Input(shape=(ENCODING_DIM, ))
decoder = Model(encoded_input, recommender.layers[-1](encoded_input))
print(prefs[prefs == 0])



def lambda_mse(frac=0.8):
    """
    Specialized loss function for recommender model.

    :param frac: Proportion of weight to give to novel ratings.
    :return: A loss function for use in a Lambda layer.
    """
    def lossfunc(xarray):
        x_in, y_true, y_pred = xarray
        zeros = tf.zeros_like(y_true)

        novel_mask = tf.not_equal(x_in, y_true)
        known_mask = tf.not_equal(x_in, zeros)

        y_true_1 = tf.boolean_mask(y_true, novel_mask)
        y_pred_1 = tf.boolean_mask(y_pred, novel_mask)

        y_true_2 = tf.boolean_mask(y_true, known_mask)
        y_pred_2 = tf.boolean_mask(y_pred, known_mask)

        unknown_loss = losses.mean_squared_error(y_true_1, y_pred_1)
        known_loss = losses.mean_squared_error(y_true_2, y_pred_2)

        # remove nans
        unknown_loss = tf.where(tf.is_nan(unknown_loss), 0.0, unknown_loss)

        return frac*unknown_loss + (1.0 - frac)*known_loss
    return lossfunc
print("function 1 is done")

def final_loss(y_true, y_pred):
    """
    Dummy loss function for wrapper model.
    :param y_true: true value (not used, but required by Keras)
    :param y_pred: predicted value
    :return: y_pred
    """
    return y_pred
print("function 2 is done")

original_inputs = recommender.input
y_true_inputs = Input(shape=(ITEM_COUNT, ))
original_outputs = recommender.output
# give 80% of the weight to guessing the missings, 20% to reproducing the knowns
loss = Lambda(lambda_mse(0.8))([original_inputs, y_true_inputs, original_outputs])

wrapper_model = Model(inputs=[original_inputs, y_true_inputs], outputs=[loss])
wrapper_model.compile(optimizer='adadelta', loss=final_loss)


def generate(pref_matrix, batch_size=64, mask_fraction=0.2):
    """
    Generate training triplets from this dataset.

    :param batch_size: Size of each training data batch.
    :param mask_fraction: Fraction of ratings in training data input to mask. 0.2 = hide 20% of input ratings.
    :param repeat: Steps between shuffles.
    :return: A generator that returns tuples of the form ([X, y], zeros) where X, y, and zeros all have
             shape[0] = batch_size. X, y are training inputs for the recommender.
    """

    def select_and_mask(frac):
        def applier(row):
            row = row.copy()
            idx = np.where(row != 0)[0]
            if len(idx) > 0:
                masked = np.random.choice(idx, size=(int)(frac*len(idx)), replace=False)
                row[masked] = 0
            return row
        return applier

    indices = np.arange(pref_matrix.shape[0])
    batches_per_epoch = int(np.floor(len(indices)/batch_size))
    while True:
        np.random.shuffle(indices)

        for batch in range(0, batches_per_epoch):
            idx = indices[batch*batch_size:(batch+1)*batch_size]
            y = np.array(pref_matrix[idx,:])
            X = np.apply_along_axis(select_and_mask(frac=mask_fraction), axis=1, arr=y)


            yield [X, y], np.zeros(batch_size)

[X, y], _ = next(generate(pref_matrix.fillna(0).values))
print(len(X[X != 0])/len(y[y != 0]))
print("# returns 0.8040994014148377")

def fit(wrapper_model, pref_matrix, batch_size=64, mask_fraction=0.2, epochs=1, verbose=1, patience=0):
    stopper = EarlyStopping(monitor="loss", min_delta=0.00001, patience=patience, verbose=verbose)
    batches_per_epoch = int(np.floor(pref_matrix.shape[0]/batch_size))

    generator = generate(pref_matrix, batch_size, mask_fraction)

    history = wrapper_model.fit_generator(
        generator,
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        callbacks = [stopper] if patience > 0 else []
    )

    return history
print("function 3 is done")

# stop after 3 epochs with no improvement
fit(wrapper_model, pref_matrix.fillna(0).values, batch_size=125, epochs=100, patience=3)
# Loss of 0.6321

def predict(ratings, recommender, mean_0, mean_i, movies):
    # add a dummy user that's seen all the movies so when we generate
    # the ratings matrix, it has the appropriate columns
    dummy_user = movies.reset_index()[["movie_id"]].copy()
    dummy_user["user_id"] = -99999
    dummy_user["rating"] = 0
    dummy_user = dummy_user.set_index(["movie_id", "user_id"])

    ratings = ratings["rating"]

    ratings = ratings - mean_0
    ratings = ratings - mean_i
    mean_u = ratings.groupby("user_id").mean()
    ratings = ratings - mean_u

    ratings = ratings.append(dummy_user["rating"])

    pref_mat = ratings.reset_index()[["user_id", "movie_id", "rating"]].pivot(index="user_id", columns="movie_id", values="rating")
    X = pref_mat.fillna(0).values
    y = recommender.predict(X)

    output = pd.DataFrame(y, index=pref_mat.index, columns=pref_mat.columns)
    output = output.iloc[1:] # drop the bad user

    output = output.add(mean_u, axis=0)

    return output
print("fucntion 4 is done")

sample_ratings = pd.DataFrame([
    {"user_id2": 1, "movie_id2": 2858, "rating": 1}, # american beauty
    {"user_id2": 1, "movie_id2": 260, "rating": 5},  # star wars
    {"user_id2": 1, "movie_id2": 480, "rating": 5},  # jurassic park
    {"user_id2": 1, "movie_id2": 593, "rating": 2},  # silence of the lambs
    {"user_id2": 1, "movie_id2": 2396, "rating": 2}, # shakespeare in love
    {"user_id2": 1, "movie_id2": 1197, "rating": 5}  # princess bride
]).set_index(["movie_id2", "user_id2"])

# predict and print the top 10 ratings for this user
#y = predict(sample_ratings, recommender, mean_0, mean_i, movies.loc[(rating_counts).index[0:500]]).transpose()
print("y is calculated")
#preds = y.sort_values(by=1, ascending=False).head(10)

#preds["title"] = movies.loc[preds.index]["title"]
print("test 1 passed")



starwars = decoder.get_weights()[0][:,33]
esb = decoder.get_weights()[0][:,144]
americanbeauty = decoder.get_weights()[0][:,401]


print(np.sqrt(((starwars - esb)**2).sum()))

