import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#USED IN PART 2 when data needed cleaning.
def clean_data(df):
    columns_with_numbers = ['rating_val', 'rating_avg', 'total_time']
    df[columns_with_numbers] = df[columns_with_numbers].fillna(0)

    strings_columns = ['category', 'cuisine', 'title', 'image_url', 'recipe_url', 'ingredients']
    df[strings_columns] = df[strings_columns].fillna('Unknown')

    return df

# Load and clean the dataset
df = pd.read_csv('recipes.csv', na_values=['', ' '])

df = clean_data(df)


features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']


def combine_features(row):
    return " ".join(str(row[feature]) for feature in features)


df['combine_features'] = df.apply(combine_features, axis=1)


count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(df['combine_features'])

# ---------------- Vector Space Method Function ----------------
def vec_space_method(recipe_title, data_frame, count_matrix):
    indices = data_frame.index[data_frame['title'] == recipe_title].tolist()
    if not indices:
        print(f"Recipe '{recipe_title}' not found.")
        return []
    target_recipe_index = indices[0]
    
    target_vector = count_matrix[target_recipe_index]
    sim_scores = count_matrix.dot(target_vector.T).toarray().ravel()
    
    sim_scores[target_recipe_index] = -1
    

    top_indices = np.argsort(sim_scores)[::-1][:10]
    

    recommendations = [
        (data_frame.iloc[i]['title'], data_frame.iloc[i]['total_time'], 
         data_frame.iloc[i]['rating_avg'], sim_scores[i])
        for i in top_indices
    ]
    return recommendations



def knn_similarity(recipe_title, data_frame, count_matrix, n_neighbors=11):

    indices = data_frame.index[data_frame['title'] == recipe_title].tolist()
    if not indices:
        print(f"Recipe '{recipe_title}' not found.")
        return []
    target_recipe_index = indices[0]
    

    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn_model.fit(count_matrix)
    

    distances, neighbor_indices = knn_model.kneighbors(count_matrix[target_recipe_index], n_neighbors=n_neighbors)
    

    distances = distances.flatten()
    neighbor_indices = neighbor_indices.flatten()
    

    neighbor_indices = neighbor_indices[1:]
    distances = distances[1:]
    

    similarities = 1 - distances
    

    recommendations = [
        (data_frame.iloc[i]['title'], data_frame.iloc[i]['total_time'], 
         data_frame.iloc[i]['rating_avg'], sim)
        for i, sim in zip(neighbor_indices, similarities)
    ]
    return recommendations


test_recipes = [
    "Chicken tikka masala",
    #The recipe in the dataset is "Albanian baked lamb with rice (Tavë kosi)"
    "Albanian baked lamb with rice (Tavë kosi)",
    "Baked salmon with chorizo rice",
    "Almond lentil stew"
]


vsm_results = {}
print("VSM Recommendations:")
for index, recipe in enumerate(test_recipes, start=1):
    suggested_recipes = vec_space_method(recipe, df, count_matrix)
    if not suggested_recipes:
        print(f"User {index} - {recipe}: Recipe not found.")
        continue
    vsm_results[recipe] = set(suggested_recipe_title for suggested_recipe_title, time, avg, similarity_score in suggested_recipes)
    print(f"User {index} - {recipe}:")
    for suggested_recipe_title, time, avg, similarity_score in suggested_recipes:
        print(f"- {suggested_recipe_title} | Total Time: {time} minutes | Rating Avg: {avg:.2f} | Score: {similarity_score:.3f}")
    print()


knn_results = {}
print("KNN Recommendations:")
for index, recipe in enumerate(test_recipes, start=1):
    suggested_recipes = knn_similarity(recipe, df, count_matrix, n_neighbors=11)
    if not suggested_recipes:
        print(f"User {index} - {recipe}: Recipe not found.")
        continue
    knn_results[recipe] = set(suggested_recipe_title for suggested_recipe_title, time, avg, similarity_score in suggested_recipes)
    print(f"User {index} - {recipe}:")
    for suggested_recipe_title, time, avg, similarity_score in suggested_recipes:
        print(f"- {suggested_recipe_title} | Total Time: {time} minutes | Rating Avg: {avg:.2f} | Score: {similarity_score:.3f}")
    print()



def compute_coverage(results_dictionary, data_frame):
    all_recommended_titles = set().union(*results_dictionary.values()) if results_dictionary else set()
    coverage_percent = (len(all_recommended_titles) / len(data_frame)) * 100
    return coverage_percent

coverage_vsm = compute_coverage(vsm_results, df)
coverage_knn = compute_coverage(knn_results, df)

all_titles = df['title'].tolist()
def build_binary_vector(rec_set):
    return [1 if title in rec_set else 0 for title in all_titles]

vsm_vectors = [build_binary_vector(vsm_results[rec]) for rec in vsm_results]
knn_vectors = [build_binary_vector(knn_results[rec]) for rec in knn_results]


def avg_pairwise_similarity(vectors):
    if len(vectors) < 2:
        return 0.0
    similarity_matrix = cosine_similarity(vectors)
    n = similarity_matrix.shape[0]
    similarity_values = [similarity_matrix[i, j] for i in range(n) for j in range(i+1, n)]
    return np.mean(similarity_values)

average_similarity_vsm = avg_pairwise_similarity(vsm_vectors)
average_similarity_knn = avg_pairwise_similarity(knn_vectors)


personalisation_vsm = 1 - average_similarity_vsm
personalisation_knn = 1 - average_similarity_knn


print(f"Coverage VSM: {coverage_vsm:.2f}%")
print(f"Personalisation VSM: {personalisation_vsm:.2f}")
print(f"\nCoverage KNN: {coverage_knn:.2f}%")
print(f"Personalisation KNN: {personalisation_knn:.2f}")


def label_tasty(rating):
    if float(rating) > 4.2:
        return 1
    else:
        return -1

df['tasty_label'] = df['rating_avg'].apply(label_tasty)

x = count_matrix  
y = df['tasty_label']

#From testing the best value for test_size was 0.41
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.41, random_state=42)

tasty_recipe_classifier = KNeighborsClassifier(n_neighbors=5)
tasty_recipe_classifier.fit(x_train, y_train)


y_pred = tasty_recipe_classifier.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nPredicted recipes that are tasty")
print(f'KNN Accuracy: %.2f' % (accuracy * 100) + '%')