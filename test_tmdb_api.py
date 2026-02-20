import os
import json
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDb_API_KEY")

movies = pd.read_csv("data/cleaned/movies.csv")

tmdb_ratings = []

# language/country of origin mapping table 
LANG_MAP = {"French": "fr", "Japanese": "ja", "English": "en", "Korean": "ko", "Spanish": "es"}
COUNTRY_MAP = {"USA": "US", "Japan": "JP", "France": "FR", "South Korea": "KR", "UK": "GB"}

for idx, row in movies.iterrows():
    movie_name = row['title']
    movie_year = str(int(row['release_year'])) if pd.notnull(row['release_year']) else ""
    current_lang = LANG_MAP.get(row['language'])
    current_country = COUNTRY_MAP.get(row['country_of_origin'])
    
    url = "https://api.themoviedb.org/3/search/multi"
    params = {"api_key": API_KEY, "query": movie_name}

    response = requests.get(url, params=params)
    
    # variables for matchings
    match_year = None
    match_country = None
    match_lang = None
    match_any = None

    if response.status_code == 200:
        results = response.json().get("results", [])
        
        # filter by movies and tv shows 
        valid_results = [r for r in results if r.get("media_type") in ["movie", "tv"]]

        for item in valid_results:
            target_date = item.get("release_date") or item.get("first_air_date")
            item_year = target_date[:4] if target_date else ""
            item_lang = item.get("original_language")
            item_countries = item.get("origin_country", [])

            # match with release year 
            if item_year == movie_year:
                match_year = item
                break # if find match, end 
            
            # match with country of origin 
            if not match_country and current_country in item_countries:
                match_country = item
                
            # match with language 
            if not match_lang and item_lang == current_lang:
                match_lang = item

        # choose by priority
        # release year > country of origin > language > title only > None
        final_match = match_year or match_country or match_lang
        
        if not final_match and valid_results:
            final_match = valid_results[0] # Level 4: match titles only

        tmdb_ratings.append(final_match.get("vote_average") if final_match else None)
    else:
        tmdb_ratings.append(None)

    if (idx + 1) % 10 == 0:
        print(f"Progress: {idx + 1}/{len(movies)} processed...")

    sleep(0.15) # to prevent the API rate limit

# store the tmdb rating column and export as csv
movies["tmdb_rating"] = tmdb_ratings
movies.to_csv("data/movies_with_external_rating.csv", index=False)