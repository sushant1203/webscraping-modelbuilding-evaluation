import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#All this answers are based on the lab in week 5

#https://www.w3schools.com/python/pandas/ref_df_fillna.asp
def clean_data(df):
    # Treating the missing values
    #This is based on the lab in week 5 question 1.
    columns_with_numbers = ['rating_val', 'rating_avg', 'total_time']
    df[columns_with_numbers] = df[columns_with_numbers].fillna(0)

    strings_columns = ['category', 'cuisine', 'title', 'image_url', 'recipe_url', 'ingredients']
    df[strings_columns] = df[strings_columns].fillna('Unknown')

    return df

#https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
#https://www.python4data.science/en/latest/clean-prep/nulls.html (Point 2 explains the na_values)
#na_values is being used to tell pandas that the empty strings or with one space are missing values 
df = pd.read_csv('recipes.csv', na_values=['', ' '])

# https://www.geeksforgeeks.org/working-with-missing-data-in-pandas/
print('\nMissing Values')
print(df.isnull().sum()) #This is to show the missing values in the dataset

df = clean_data(df)

# Now no missing values remain
print('\nMissing Values After Cleaning:')
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

_10_highest_rated_recipes = df.sort_values(by='rating_val', ascending=False).head(10)
print("\nTop 10 highest rated recipes:")
print(_10_highest_rated_recipes)


# QUESTION 2


top_10_highest_average_ratings = df.groupby('title')['rating_avg'].mean().reset_index().sort_values(by='rating_avg', ascending=False).head(10)
print("\nTop 10 recipes by average rating:")
print(top_10_highest_average_ratings)


num_bootstrap_samples = 1000


bootstrap_samples = [np.random.choice(df['rating_avg'], size=100, replace=True) for _ in range(num_bootstrap_samples)]

bootstrap_means = [np.mean(sample) for sample in bootstrap_samples]
confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
print("\n95% confidence interval for the average recipe rating: [{:.2f}, {:.2f}]".format(confidence_interval[0], confidence_interval[1]))


#Question 3

plt.figure(figsize=(12, 8))
plt.scatter(df['rating_val'], df['rating_avg'])
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.grid()
plt.show()


# Suggested Threshold:
# - As known all the ratings matter, even thought recipes with low number of ratings can be unreliable, all the recipes
# start with a low number of ratings and then they get more ratings. So, the threshold should be 0.
# In another scenario we can consider a treshhold of 5 or 10 ratings but in this case we are not going to do that. When a recipe
# has more ratings, it is more reliable. Taking into consideration recipes with low number of ratings can completely change
# recommendations based on ratings but it is still valid to say that the recipe is highly rated.
# To summarize, the threshold should be 0.



#Question 4


features = ['title','rating_avg','rating_val','total_time','category','cuisine','ingredients']

def combine_features(row):
    return " ".join(str(row[feature]) for feature in features)

df['combine_features'] = df.apply(combine_features, axis=1)
print("\nFirst 5 rows of 'combine_features':")
print(df[['title', 'combine_features']].head())


count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(df['combine_features'])
cosine_sim = cosine_similarity(count_matrix)
print("\nShape of cosine similarity matrix:", cosine_sim.shape)


liked_recipe = "Chicken and coconut curry"
liked_indices = df[df['title'] == liked_recipe].index.tolist()

if not liked_indices:
    print(f"\nRecipe '{liked_recipe}' not found in the dataset.")
else:
    liked_index = liked_indices[0]
    sim_scores = cosine_sim[liked_index]
    

    sim_scores_index = list(enumerate(sim_scores))
    sim_scores_index = sorted(sim_scores_index, key=lambda x: x[1], reverse=True)
    
    #skips the first element since it's the recipe itself and then gets the next 10
    top_10 = sim_scores_index[1:11]
    
    print(f"\nTop 10 recipe recommendations for '{liked_recipe}':")
    for index, score in top_10:
        print(f"- {df.iloc[index]['title']} (similarity: {score:.3f})")
