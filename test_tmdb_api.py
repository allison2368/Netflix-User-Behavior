import requests
import pandas as pd
import json
from time import sleep
import os

API_KEY = os.getenv("API_KEY")

# movie_name = "Squid Game"

# url = "https://api.themoviedb.org/3/search/movie"
# params = { "api_key": API_KEY, "query": movie_name, "language": "ko-KR" }

# print("--- API connection test ---")
# response = requests.get(url, params=params)

# if response.status_code == 200:
#     data = response.json()
#     if data["results"]: 
#         first_movie = data["results"][0]
#         print("successful data pull!")
#         print("title:", first_movie["title"])
#         print("ratings:", first_movie["vote_average"]) 
#         print("released date:", first_movie["release_date"]) 
#     else: 
#         print("no result.") 
# else: 
#     print("error:", response.status_code)


movies = pd.read_csv("data/movies.csv")

tmdb_ratings = []

for idx, row in movies.iterrows():
    movie_name = row['title']
    movie_year = row['release_year']
    
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": API_KEY,
        "query": movie_name,
        "year": movie_year 
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            # if release_date is not null, compare only with year
            matched = None
            for movie in data["results"]:
                release_date = movie.get("release_date", "")
                if release_date and release_date[:4] == str(movie_year):
                    matched = movie
                    break
            if matched is None:
                matched = data["results"][0]  # fallback
            
            tmdb_ratings.append(matched["vote_average"])
        else:
            tmdb_ratings.append(None)
    else:
        print("Error:", response.status_code)
        tmdb_ratings.append(None)
    
    sleep(0.25)  # to prevent the API rate limit

movies["tmdb_rating"] = tmdb_ratings
print(movies)

movies.to_csv("data/movies_with_tmdb_rating.csv", index=False)
