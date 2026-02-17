# ; import pandas as pd

# ; # Load data
# ; movies = pd.read_csv("data/cleaned/movies_cleaned.csv")

# ; print(f"가져온 영화 개수: {len(movies)}") # 1000개 확인!

# ; # URL 구성 예시
# ; url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={movie_name}&y={movie_year}"
# ; response = requests.get(url)
# ; data = response.json()

# ; # 로튼 토마토 점수만 쏙 뽑기
# ; rt_score = None
# ; if "Ratings" in data:
# ;     for rating in data["Ratings"]:
# ;         if rating["Source"] == "Rotten Tomatoes":
# ;             rt_score = rating["Value"]



# ; import os
# ; import pandas as pd
# ; import requests
# ; from dotenv import load_dotenv
# ; from time import sleep

# ; load_dotenv()
# ; TMDB_KEY = os.getenv("TMDB_API_KEY")
# ; OMDB_KEY = os.getenv("OMDB_API_KEY")

# ; # eda.py에서 전처리된 데이터 불러오기 (방법 1 사용 시)
# ; from eda import movies 

# ; ratings = []

# ; for idx, row in movies.iterrows():
# ;     title = row['title']
# ;     year = row['release_year']
    
# ;     # 1단계: TMDb 검색 (multi 검색 활용)
# ;     tmdb_url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_KEY}&query={title}"
# ;     res = requests.get(tmdb_url).json()
    
# ;     final_rating = None
# ;     if res.get('results'):
# ;         # 연도 매칭 시도 (첫 번째 결과)
# ;         final_rating = res['results'][0].get('vote_average')
        
# ;     # 2단계: TMDb 실패 시 OMDb 출동!
# ;     if not final_rating:
# ;         print(f"TMDb 실패 -> OMDb 시도: {title}")
# ;         omdb_url = f"http://www.omdbapi.com/?apikey={OMDB_KEY}&t={title}&y={year}"
# ;         omdb_res = requests.get(omdb_url).json()
        
# ;         if omdb_res.get('Response') == 'True':
# ;             # OMDb의 imdbRating을 가져오거나 Rotten Tomatoes 점수 추출
# ;             final_rating = omdb_res.get('imdbRating')

# ;     ratings.append(final_rating)
# ;     sleep(0.1) # 속도 조절

# ; movies['combined_rating'] = ratings
# ; movies.to_csv("data/movies_final_ratings.csv", index=False)


import os
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

load_dotenv()

# 환경 변수에서 키 가져오기
OMDb_API_KEY = os.getenv("OMDb_API_KEY")

# 전처리된 1,000개 데이터 로드
movies = pd.read_csv("data/movies_with_external_rating.csv")

imdb_ratings = []
rt_ratings = []

print(f"총 {len(movies)}개의 데이터를 처리하기 시작합니다. (OMDb 1,000회 제한 주의)")

for idx, row in movies.iterrows():
    movie_name = row['title']
    movie_year = row['release_year']
    
    # OMDb API는 t(제목), y(연도)를 넣으면 영화/TV 가리지 않고 가장 잘 맞는 하나를 찾아줌
    url = "http://www.omdbapi.com/"
    params = {
        "apikey": OMDb_API_KEY,
        "t": movie_name,
        "y": movie_year
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("Response") == "True":
                # 1. IMDb 평점 가져오기
                imdb_ratings.append(data.get("imdbRating"))
                
                # 2. Rotten Tomatoes 점수 찾기 (Ratings 리스트 안에 있음)
                rt_score = None
                for r in data.get("Ratings", []):
                    if r["Source"] == "Rotten Tomatoes":
                        rt_score = r["Value"]
                        break
                rt_ratings.append(rt_score)
            else:
                # 검색 결과가 없는 경우
                imdb_ratings.append(None)
                rt_ratings.append(None)
        else:
            print(f"Error {response.status_code} at index {idx}")
            imdb_ratings.append(None)
            rt_ratings.append(None)
            
    except Exception as e:
        print(f"Request failed at index {idx}: {e}")
        imdb_ratings.append(None)
        rt_ratings.append(None)

    # 진행 상황 출력 (100개마다)
    if (idx + 1) % 100 == 0:
        print(f"진행 상황: {idx + 1}/1000 완료...")

    # OMDb는 초당 호출 제한이 널널하지만 안전을 위해 약간의 대기
    sleep(0.1)

# 데이터프레임에 추가
movies["imdb_rating_omdb"] = imdb_ratings
movies["rotten_tomatoes_omdb"] = rt_ratings

# 최종 결과 저장
movies.to_csv("data/movies_with_all_ratings.csv", index=False)
print("성공적으로 저장되었습니다: data/movies_with_all_ratings.csv")