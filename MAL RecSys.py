# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:13:06 2020

@author: Jia
"""
import time
start_time = time.time()

# importing relevant libraries
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import SVD, NMF, SlopeOne
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split


from RecSysCompare import RecSysCompare

print('Loading files...')
#load MAL anime data
MAL_anime = pd.read_csv('anime.csv', names = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'], skiprows=1)
MAL_anime.dropna(inplace=True)
MAL_series = MAL_anime[MAL_anime['type'] == 'TV']

#load MAL rating data for first 2000 users
MAL_predict = pd.read_csv('rating.csv', names = ['user_id', 'anime_id', 'user_rating'], skiprows=1)
MAL_rating = pd.read_csv('rating.csv', names = ['user_id', 'anime_id', 'user_rating'], skiprows=1, na_values=-1)
MAL_rating.dropna(inplace=True)



print('Filtering dataset...')
MAL_predict = MAL_series.merge(MAL_predict, how='left', left_on=['anime_id'], right_on=['anime_id'])
MAL_merged = MAL_series.merge(MAL_rating, how='left', left_on=['anime_id'], right_on=['anime_id'])

# Filter anime to those that have been rated at least 1000 times (aka by at least ~1.36% of userbase)
min_anime_ratings = 1000
filter_anime = MAL_merged['anime_id'].value_counts() > min_anime_ratings
filter_anime = filter_anime[filter_anime].index.tolist()

# Filter users to those that have rated at least 200 times (this gives us ~11.7% of the userbase)
# To find individual users, try this: https://myanimelist.net/comments.php?id=42635
min_user_ratings = 100
filter_users = MAL_merged['user_id'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

# This is the workable dataset that applies both user/anime filters, Reader specifies the rating scale used
MAL_filtered = MAL_merged[MAL_merged['anime_id'].isin(filter_anime) & MAL_merged['user_id'].isin(filter_users)]
MAL_predict = MAL_predict[MAL_predict['anime_id'].isin(filter_anime) & MAL_predict['user_id'].isin(filter_users)]
data = Dataset.load_from_df(MAL_filtered[['user_id','anime_id','rating']], Reader(rating_scale=(0,10)))

## Testing different algorithms to see which one has lowest rmse
#RecSysCompare(data, SVD(biased=False), NMF(), SlopeOne())
## Results: SVD provides best results/time.
## Surprisingly KNN-Baseline is faster and more accurate than SVDpp, but is it a case of overfitting?

# Decision: implement recommender system with SVD
algo = SVD(biased=False)

# Cross Validation and implementing SVD
print('Cross validation')
cross_validate(algo, data, measures = ['RMSE'], cv=5, verbose=False)
print('Creating train-test sets')
trainset, testset = train_test_split(data, test_size = 0.2)
print('implementing SVD algorithm')
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)

print('Making predictions')
data_predict = []
for row in MAL_predict[MAL_predict['user_rating'] == -1].itertuples():
    temp_predict = algo.predict(row[8], row[1])
    rating_diff = row[6] - round(temp_predict[3], 2)
    data_predict.append((temp_predict[0], temp_predict[1], round(temp_predict[3], 2), rating_diff))
data_predict = pd.DataFrame(data_predict, columns=['User ID','Anime ID','Predicted Rating', 'Rating Difference'])

# Sort by user ID
data_predict.sort_values('User ID')
temp = MAL_filtered[MAL_filtered['user_id'] < 10000]
temp.sort_values('user_id')
data_predict.to_csv('final_predictions.csv', index=False)
temp.to_csv('MAL_filtered.csv', index=False)


print("--- %s seconds ---" % (time.time() - start_time))

