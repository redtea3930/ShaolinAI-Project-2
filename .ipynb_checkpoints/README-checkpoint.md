# Comparing Critic and Audience Reviews
### Athen, Donny, and Jim; Oct 5, 2023

***
## Intro
Entertainment review sites frequently show a disparity between critic and audience reviews of movies and shows. This project seeks to find features of those critic reviews that diverge from average audience score of a media product.  

***
## Source Data
In order to make this analysis, we determined we would need a dataset containing at least an average audience score, an average critic score, and a critic review text associated to a large amount of media titles. After some searching we were able to find this:

[Massive Rotten Tomatoes Movies & Reviews](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?select=rotten_tomatoes_movie_reviews.csv), Andrea Villa, uploaded to Kaggle 2023

This is indeed a "massive" dataset, containing some 1.4 million movie reviews. It consists of two csv files, listed here with their data columns:

* rotten_tomatoes_movies.csv: id, title, audienceScore, tomatoMeter, rating, releaseDateTheaters, releaseDateStreaming, runtimeMinutes, genre, director, and boxOffice revenue.

* rotten_tomatoes_movie_reviews.csv: id, reviewId, creationDate, criticName, isTopCritic, originalScore, reviewState, publicationName, reviewText, scoreSentiment, and reviewUrl.

These csv's had all the data we needed between them, in these columns:
* average audience score: audienceScore
* average critic score: tomatoMeter
* critic review text: reviewText
* media title: title
***
## Data Cleaning

* Our first step was to do an inner merge on the two csv's to have all input data within one dataframe. We next created a column for the disparity between critic and audience scores for each movie:
>     massive['delta'] = massive['tomatoMeter'] - massive['audienceScore']

* We then dropped columns that were unlikely to contribute meaningful data to a machine learning model, including the input audience score and critic score used to create 'delta':

>     massive = massive.drop(columns=['originalScore', 'rating', 'ratingContents', 'releaseDateTheaters',
>                                'releaseDateStreaming', 'runtimeMinutes', 'genre', 'originalLanguage',
>                                'director', 'writer', 'boxOffice', 'distributor', 'soundMix', 'reviewUrl',
>                                'id', 'reviewId', 'creationDate', 'isTopCritic', 'reviewState', 'tomatoMeter',
>                                'audienceScore'])


* There were several nulls, all of which were in the four specified data columns and could not be reasonably replaced; all these nulls were dropped. 


* As we are focusing on movies with a large discrepancy between critic and audience scores, we selected reviews where this discrepancy was at least 20% either above or below the audience score:


>     condition = (massive['delta'] >= 20) & (massive['scoreSentiment'] == 'POSITIVE')
>          if condition.any():
>          massive = massive.drop(massive[condition].index)
>     condition = (massive['delta'] <= -20) & (massive['scoreSentiment'] == 'NEGATIVE')
>          if condition.any():
>          massive = massive.drop(massive[condition].index)

* Lastly, we considered that there may be duplicated reviews in the remaining data. To be sure to get rid of these, we dropped duplicates in the 'reviewText' column:

>     massive = massive.drop_duplicates(subset='reviewText', keep='first')
***
## Data Processing

* Despite data cleaning, we had difficulty working with this large dataset, so we took a ra

***
## Conclusions

***
## Challenges
