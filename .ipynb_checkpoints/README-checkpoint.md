# Comparing Critic and Audience Reviews
### Athen, Donny, and Jim; Oct 5, 2023

***
## Intro
Entertainment review sites frequently show a disparity between critic and audience reviews of movies and shows. This project seeks to find features of those critic reviews that diverge from average audience score of a media product.  

***
## Source Data
In order to make this analysis, we determined we would need a dataset containing at least an average audience score, an average critic score, and a critic review text associated to a large amount of media titles. After some searching we were able to find this:

[Massive Rotten Tomatoes Movies & Reviews](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?select=rotten_tomatoes_movie_reviews.csv), uploaded to Kaggle by Andrea Villa, 2023

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

Our first step was to do an inner merge on the two csv's to have all input data within one dataframe. We then dropped columns that were unlikely to contribute meaningful data to a machine learning model: 'originalScore', 'rating','ratingContents', 'releaseDateTheaters','releaseDateStreaming', 'runtimeMinutes', 'genre', 'originalLanguage', 'director', 'writer', 'boxOffice', 'distributor', 'soundMix', 'reviewUrl'. There were several nulls, all of which were in the four specified data columns and could not be reasonably replaced; all these nulls were dropped. 

***
## Data Processing

***
## Conclusions

***
## Challenges
