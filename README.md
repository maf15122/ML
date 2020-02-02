Project Title: Application of Machine Learning in Recommender Systems

Contributors: Mahdi and Anchala  

In this project we will answer the following questions:

1-	By considering IMDb movie dataset, what are the 5 most similar movies to a movie search?

2-	By using KNN algorithm, how does considering different distance metrics (e.g. Euclidian, Mahalanobis, and Manhattan) affect the results? 


This project includes the following folders:

1)Literature; it includes all of the articles that are used for this project

2)Data; it contains various dataset from IMDb 

3)Code; this project will be written in python and will be saved in this folder

References:

1) https://www.kdnuggets.com/2019/04/building-recommender-system.html: Walks us through building a simple recommender systerm, including some basic python code, using Movielense data. One way to produce a list of similar movies for cataegorical data is called cosine similarity. For any two items i and j, the cosine similarity of i and j is simply the cosine of the angle between i and j where i and j are interpreted as vectors in feature space. So basically, cosine similarity of 1 implies having all the same genre tags. Another way of producing a list of similar films is by using simple collaborative filtering method. Collaborative filtering recommends items based on what similar users liked. This can be done using the ratings the other viewers provide. 

2) https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761: A step by step Python code of KNN algorithm

3) https://datasets.imdbws.com/: IMDb dataset.

4) https://builtin.com/data-science/recommender-systems: This article starts out with an explaination of what recommender systems are and then walks us through different types of similarity measure; Minkowski Distance is used when the data points are numerical, Manhattan Distance is the distance between two points measured along axes at right angles, Euclidean distance or the L2 norm, Cosine similarity, Pearson Coefficient is the measure of correlation between two random variables and ranges between [-1,1], Jaccard Similarity is used where the objects are points or vectors, and Hamming Distance is used for categorical variables.

5) https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7452337: An article about Memory-based recommender systems including algorithm.
