import pandas as pd

# print all rows and column 
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None) 

# Load data
tables = {
    "users": ("data/users.csv", "user_id"),
    "watch": ("data/watch_history.csv", "session_id"),
    "ser_logs": ("data/search_logs.csv", "search_id"),
    "reviews": ("data/reviews.csv", "review_id"),
    "rec_logs": ("data/recommendation_logs.csv", "recommendation_id"),
    "movies": ("data/movies.csv", "movie_id"),
}

# dictionary for storing DFs
dfs = {}

print("===== Duplicate Check Before Cleaning =====")

for name, (path, pk) in tables.items():
    df = pd.read_csv(path)
    dfs[name] = df
    
    print(f"\n[{name}]")
    print("Total rows:", len(df))
    print("Unique PK:", df[pk].nunique())
    print("Exact duplicate rows:", len(df) - len(df.drop_duplicates()))

# ------------------------------
# Remove duplicates
# ------------------------------

print("\n===== Removing Duplicates =====")

for name, (_, pk) in tables.items():
    df = dfs[name]
    df_clean = df.drop_duplicates(subset=pk, keep='first')
    dfs[name] = df_clean
    
    print(f"{name}: {len(df)} → {len(df_clean)}")

# ------------------------------
# Final Check
# ------------------------------

print("\n===== Final Duplicate Validation =====")

for name, (_, pk) in tables.items():
    df = dfs[name]
    print(f"{name} duplicated PK count:",
          df[pk].duplicated().sum())


users = dfs["users"]
watch = dfs["watch"]
ser_logs = dfs["ser_logs"]
reviews = dfs["reviews"]
rec_logs = dfs["rec_logs"]
movies = dfs["movies"]

print("\nSample movies:")
print(movies.head())