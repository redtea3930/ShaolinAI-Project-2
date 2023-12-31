# Comparing Critic and Audience Reviews
### Athen, Donny, and Jim; Oct 5, 2023

***
***
## Intro
Entertainment review sites frequently show a disparity between critic and audience reviews of movies and shows. This project seeks to find features of those critic reviews that diverge from average audience score of a media product.  

***
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
***
## Data Cleaning

* Our first step was to do an inner merge on the two csv's to have all input data within one dataframe. We next created a column for the disparity between critic and audience scores for each movie:
>     massive['delta'] = massive['tomatoMeter'] - massive['audienceScore']

* We then dropped columns that were unlikely to contribute meaningful data to a machine learning model, including the input audience score and critic score used to create 'delta':
```
massive = massive.drop(columns=['originalScore', 'rating', 'ratingContents', 'releaseDateTheaters',
                                'releaseDateStreaming', 'runtimeMinutes', 'genre', 'originalLanguage',
                                'director', 'writer', 'boxOffice', 'distributor', 'soundMix','reviewUrl',
                                'id', 'reviewId', 'creationDate', 'isTopCritic', 'reviewState', 'tomatoMeter',
                                'audienceScore'])
```

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

* This provided our final cleaned dataframe:
![massive_cleaned](output_plots/massive_cleaned.png)

***
***
## Data Preprocessing
### Random subsampling
* Even after cleaning, our dataframe still contained 965167 rows. In order to be able to process this at all we first took a random subsample of the cleaned data:
>     massive = massive.sample(n=15000)

### Tokenization
* We also needed to tokenize the text of the review column to perform NLP analysis. Our process for this consisted of removing "stop words" ie very commonly used words that provide little useful data, removing anything containing non-alphabetic characters, setting all words to a base gramatical form (lemma), and encoding the text of reviews as a sequence of tokens:
```
     lemmatizer = WordNetLemmatizer()
     import re
     def process_text(text): 
          sw = set(stopwords.words('english')) 
          regex = re.compile("[^a-zA-Z ]") 
          re_clean = regex.sub('', text) 
          words = word_tokenize(re_clean) 
          lem = [lemmatizer.lemmatize(word) for word in words] 
          output = ' '.join([word.lower() for word in lem if word.lower() not in sw]) 
    return output
    
    massive['reviewText'] = massive['reviewText'].apply(lambda x: process_text(x))
```

### Limiting populations of critics and publications
*  As shown, the number of reviews per critic and per publication were sharply distributed. We therefore decided to sort the right tail of critic and publication distributions into an "other" category before encoding these categorical data columns in order to limit the number of dummy values in models using these columns. Thresholds for the minimum number of reviews to set critics and publications as their own category were manually selected.

![Publications](output_plots/histo_publicatioName.png)

![Critics](output_plots/histo_criticName.png)

```
counts = combined.criticName.value_counts()
threshold = combined.criticName.isin(counts.index[counts<16])
combined.loc[threshold, 'criticName'] = 'Other'
```

```
counts = combined.publicatioName.value_counts()
threshold = combined.publicatioName.isin(counts.index[counts<12])
combined.loc[threshold, 'publicatioName'] = 'Other'
```
***
## Modeling
## TF-IDF Sentiment Model
Term Frequency-Inverse Document Frequency (TF-IDF) is a measure combining how frequently a term appears within a document (Term Frequency) with the importance of a term within a corpus of documents (Inverse Document Frequency) to assign a weight to each term in a document. Our first model attempts to predict the critic score sentiment, whether 'positive' or 'negative', based on a TF-IDF of the critic reviews. 
### Process
* Fit TFIDF vectorizer for sentiment model
* Split the data into training and testing sets
* Train Logistic Regression model


### Results
```
    model.score(X_train, y_train)
    0.8237333333333333
    model.score(X_test, y_test)    
    0.76
```
***
## TF-IDF on 'delta' (critic-audience score discrepancy)

### Process

* Rename 'title' column to 'title_' to prevent confusion with instances of the word "title" in vectorized or dummy columns

* Vectorizing original 'reviewText' to dense array for linear model and combine with original dataframe, drop 'reviewText' now that vectorized words are all columns
```
tfidf_dense = tfidf_vectorizer.fit_transform(massive['reviewText']).todense()
new_cols = tfidf_vectorizer.get_feature_names()
combined = massive.join(pd.DataFrame(tfidf_dense, columns=new_cols))

combined = combined.drop(columns=['reviewText'])
```
* Limit populations of critics and publications
```
counts = combined.criticName.value_counts()
threshold = combined.criticName.isin(counts.index[counts<16])
combined.loc[threshold, 'criticName'] = 'Other'

counts = combined.publicatioName.value_counts()
threshold = combined.publicatioName.isin(counts.index[counts<12])
combined.loc[threshold, 'publicatioName'] = 'Other'
```
* Encode dummy values for categorical data columns: 'title_', 'criticName', 'publicatioName', 'scoreSentiment'
```
categorical_cols = ['title_', 'criticName', 'publicatioName', 'scoreSentiment']
combined = pd.get_dummies(combined, columns = categorical_cols)
```
* Set 'delta' as target and remaining columns as X
```
X = combined.drop(columns=['delta'])
y = combined['delta']
```

* Split the data into training and testing sets
>     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

* Scale X
* Perform Principle Component Analysis (PCA)
* Train Linear Regression model
```
model = LinearRegression()
model.fit(X_train, y_train)
```
### Results
***
## BERT model on 'delta' (critic-audience score discrepancy)
Bidirectional Encoder Representations from Transformers (BERT) is an open source natural language processing model developed by Google AI and released in 2018. 
* Set tokenizer and model
* define embeddings
* define vectors
* Encode dummy values for categorical data columns: 'title_', 'criticName', 'publicatioName', 'scoreSentiment'
* Set 'delta' as target and remaining columns as X
* Split the data into training and testing sets
* Scale X
* Perform Principle Component Analysis (PCA)
* Choose a machine learning model (e.g., Logistic Regression) and train it
* Results:
***
## Incorporating Gradient Boosting

***
***


***
***
## Conclusions
