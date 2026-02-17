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

for idx, row in movies.iterrows():
    movie_name = row['title']
    movie_year = str(row['release_year'])
    
    # Use search/multi (include movie + TV shows both)
    url = "https://api.themoviedb.org/3/search/multi"
    params = {
        "api_key": API_KEY,
        "query": movie_name,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        
        matched = None
        if results:
            for item in results:
                # Use release_date for movies, first_air_date for TV shows 
                target_date = item.get("release_date") or item.get("first_air_date")
                
                if target_date and target_date[:4] == movie_year:
                    matched = item
                    break
            
            # If cannot find matching year, just use the first result 
            if not matched:
                matched = results[0]
            
            tmdb_ratings.append(matched.get("vote_average"))
        else:
            tmdb_ratings.append(None)
    sleep(0.25)  # to prevent the API rate limit

movies["tmdb_rating"] = tmdb_ratings
print(movies)

movies.to_csv("data/movies_with_external_rating.csv", index=False)
