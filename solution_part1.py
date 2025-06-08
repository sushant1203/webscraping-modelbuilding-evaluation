import requests              
from bs4 import BeautifulSoup 
import pandas as pd         
import json
import isodate

#support links:
# https://scrapeops.io/python-web-scraping-playbook/python-beautifulsoup-find/
# https://www.youtube.com/watch?v=A1s1aGHoODs
# https://www.ming2k.com/python_cheat_sheet.pdf
# https://www.youtube.com/watch?v=nBzrMw8hkmY
# https://www.youtube.com/watch?v=9N6a-VLBa2I
# https://www.youtube.com/watch?v=XVv6mJpFOb0

# https://www.youtube.com/watch?v=QNLBBGWEQ3Q
# https://developers.google.com/search/docs/appearance/structured-data/intro-structured-data
# https://www.youtube.com/watch?v=DcI_AZqfZVc


def iso_to_minutes(duration_str):
    try:
        duration = isodate.parse_duration(duration_str)
        # Convert total seconds to minutes.
        return int(duration.total_seconds() / 60)
    except Exception as e:
        return 0

#recipe_url
def collect_page_data(url):
    #request.get(url) Makes a request to a web page
    page = requests.get(url)

    #Based on lab with the Supermarkets
    soup = BeautifulSoup(page.text, 'html.parser')

    #As mentioned by the teacher during lab 7 if we inspect element, go to <head> we can find all the information needed under <script data-rh="true" type="application/ld+json"> therefore we are going to be using JSON LD
    # Another to get this information would be: script = soup.find_all('script')[0].text.strip() as we visualized in one of the videos under #support links
    script = soup.find('script', type='application/ld+json')

    # https://www.geeksforgeeks.org/json-loads-in-python/
    # Got this json.loads from different youtube videos which are referenced in the support links
    json_data = json.loads(script.string) #Converts the string into a dictionary


    #https://www.w3schools.com/python/python_ref_dictionary.asp
    # The following was used to format our JSON: https://jsonformatter.org/
    recipe_data = None
    if '@graph' in json_data: #check if json has @graph
        for item in json_data['@graph']: 
            if item.get('@type') == 'Recipe':
                recipe_data = item
                break
    else:
        if json_data.get('@type') == 'Recipe': #This is in case the json does not have @graph it will still look for @type to match Recipe
            recipe_data = json_data


    #https://www.w3schools.com/python/python_ref_dictionary.asp
    # title collum
    title = recipe_data.get('name', 'N/A')

    # total_time collum
    prepTime = recipe_data.get('prepTime', '')
    cookTime = recipe_data.get('cookTime', '')
    total = iso_to_minutes(prepTime) + iso_to_minutes(cookTime)
    total_time = f'{total} minutes'


    # "image": {
    #     "@type": "ImageObject",
    #     "width": 1600,
    #     "height": 900,
    #     "url": "https://ichef.bbci.co.uk/food/ic/food_16x9_1600/recipes/avocado_pasta_with_peas_31700_16x9.jpg"
    #   }
    # Basically from image we going to get the url
    image = recipe_data.get('image', {}).get('url', 'N/A')

    ingredients = recipe_data.get('recipeIngredient', [])
    ingredients = '\n'.join(ingredients) if ingredients else 'N/A'

    rating_val = recipe_data.get('aggregateRating', {}).get('ratingValue', 'N/A')

    rating_count = recipe_data.get('aggregateRating', {}).get('ratingCount', 'N/A')

    category = recipe_data.get('recipeCategory', 'N/A')

    cuisine = recipe_data.get('recipeCuisine', 'N/A')

    diet = recipe_data.get('suitableForDiet', [])
    diet = '\n'.join(diet) if diet else 'N/A'

    # "suitableForDiet": [
    #     "http://schema.org/LowLactoseDiet",
    #     "http://schema.org/LowCalorieDiet",
    #     "http://schema.org/VeganDiet",
    #     "http://schema.org/VegetarianDiet"
    #   ]

    # in diet because we mentioned that diet is suitable for diet 
    vegan = 'Vegan' if 'http://schema.org/VeganDiet' in diet else 'Not Vegan'

    vegetarian = 'Vegetarian' if 'http://schema.org/VegetarianDiet' in diet else 'Not Vegetarian'

    recipe_dictionary = {
        'title': title,
        'total_time': total_time,
        'image': image,
        'ingredients': ingredients,
        'rating_val': rating_val,
        'rating_count': rating_count,
        'category': category,
        "cuisine": cuisine,
        'diet': diet,
        'vegan': vegan,
        'vegetarian': vegetarian,
        'url': url
    }

    return recipe_dictionary




multiple_recipes_urls = [
    'https://www.bbc.co.uk/food/recipes/avocado_pasta_with_peas_31700',
    'https://www.bbc.co.uk/food/recipes/ackee_and_saltfish_with_73421',
    'https://www.bbc.co.uk/food/recipes/marry_me_chicken_pasta_65627',
    'https://www.bbc.co.uk/food/recipes/spaghettiallacarbona_73311',
    'https://www.bbc.co.uk/food/recipes/smoked_salmon_courgette_49013',
    'https://www.bbc.co.uk/food/recipes/vegan_mushroom_80921',
    'https://www.bbc.co.uk/food/recipes/rib-eye_steak_with_61963',
    'https://www.bbc.co.uk/food/recipes/pappardelle_with_duck_31170',
    'https://www.bbc.co.uk/food/recipes/baked_sea_bream_28386',
    'https://www.bbc.co.uk/food/recipes/king_prawn_and_scallop_49681',
    'https://www.bbc.co.uk/food/recipes/pork_chop_maman_blanc_71608',
    'https://www.bbc.co.uk/food/recipes/bang_bang_cauliflower_38036',
    'https://www.bbc.co.uk/food/recipes/steak_diane_with_saut_67797',
    'https://www.bbc.co.uk/food/recipes/chicken_fricassee_49365',
    'https://www.bbc.co.uk/food/recipes/kleftiko_27007'
]


recipes = [] #This is going to store all recipes
for url in multiple_recipes_urls:
    try:
        recipe = collect_page_data(url)
        recipes.append(recipe)
        print(f"Successfully scraped {url}")
    except Exception as e:
        print(f"Failed to scrape {url}")


# https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe

df = pd.DataFrame(recipes)

# https://stackoverflow.com/questions/16923281/writing-a-pandas-dataframe-to-csv-file

df.to_csv('recipe_scrapped.csv', index=False)
