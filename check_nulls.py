import pandas as pd

# load fiile
file_path = "data/movies_with_external_rating.csv"
df = pd.read_csv(file_path)

# total rows count
total_count = len(df)

# count the number of rows of null tmdb_rating
null_count = df["tmdb_rating"].isnull().sum()

# calculate null percentage
null_percentage = (null_count / total_count) * 100

print("-" * 30)
print(f"Total number of movies: {total_count}")
print(f"The number of null ratings in TMDb: {null_count}")
print(f"Null value percentage: {null_percentage:.2f}%")
print("-" * 30)

# Extract rows that tmdb_rating is null
missing_ratings = df[df["tmdb_rating"].isnull()]

print(f"--- No tmdb ratings movies/show (Total {len(missing_ratings)} counts) ---")


pd.set_option('display.max_rows', None)  # Show all missing ratings movie/show
columns_to_show = ["movie_id", "title", "release_year", "language", "country_of_origin"]

print(missing_ratings[columns_to_show])

missing_ratings.to_csv("data/missing_tmdb_ratings.csv", index=False)