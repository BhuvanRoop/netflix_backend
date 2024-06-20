import requests
import pandas as pd
import pickle
# import nltk


url1="https://api.themoviedb.org/3/movie/popular?api_key=123bfda0107841fdca36072a39a2dcb7&language=en-US&page=1"
url2="https://api.themoviedb.org/3/movie/upcoming?api_key=123bfda0107841fdca36072a39a2dcb7&language=en-US&page=1"
url3="https://api.themoviedb.org/3/movie/top_rated?api_key=123bfda0107841fdca36072a39a2dcb7&language=en-US&page=1"

response1=requests.get(url1)
response2=requests.get(url2)
response3=requests.get(url3)

titles=[]
overviews=[]
genre_ids=[]
for i in range(len(response1.json()['results'])):
    titles.append(response1.json()['results'][i]['title'])
    overviews.append(response1.json()['results'][i]['overview'])
    genre_ids.append(response1.json()['results'][i]['genre_ids'])
    
for i in range(len(response2.json()['results'])):
    titles.append(response2.json()['results'][i]['title'])
    overviews.append(response2.json()['results'][i]['overview'])
    genre_ids.append(response2.json()['results'][i]['genre_ids'])
    
for i in range(len(response3.json()['results'])):
    titles.append(response3.json()['results'][i]['title'])
    overviews.append(response3.json()['results'][i]['overview'])
    genre_ids.append(response3.json()['results'][i]['genre_ids'])
    

df=pd.DataFrame()
df['titles']=titles
df['overviews']=overviews
df['genre_ids']=genre_ids
df.drop_duplicates(subset=['titles'],inplace=True)
df. reset_index(inplace = True, drop = True)
# df

def genre_similarity(list1,list2):
    count=0
    not_count=0
    if len(list1)<=len(list2):
        for i in range(len(list1)):
            if list1[i] in list2:
                count+=1
            else:
                not_count+=1
        not_count+=len(list2)-count

    else:
        for i in range(len(list2)):
            if list2[i] in list1:
                count+=1
            else:
                not_count+=1
        not_count+=len(list1)-count
    
    return count/(count+not_count)

overall_sim=[]
for i in range(len(df['genre_ids'])):
            sim=[]
            list1=df['genre_ids'][i]
            for j in range(len(df['genre_ids'])):
                              list2=df['genre_ids'][j]
                              sim.append(genre_similarity(list1,list2))
            overall_sim.append(sim)
                           
df['genre_sim']=overall_sim

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=50,stop_words='english')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

df['overviews']=df['overviews'].apply(stem)
vectors=cv.fit_transform(df['overviews']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
overview_sim=cosine_similarity(vectors)

overall_similarity=[]
for i in range(len(overview_sim)):
    sim1=overview_sim[i]
    sim2=df['genre_sim'][i]
    sim=[]
    for j in range(len(sim1)):
        num=(sim1[j]+sim2[j])/2
        sim.append(num)
    overall_similarity.append(sim)       

df['overall_similarity']=overall_similarity

def recommend(movie):
    movie_index=df[df['titles']==movie].index[0]
    distances=overall_similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(df.iloc[i[0]].titles)

# print(df)
# print(df.columns)
# print(overall_similarity)
# recommend('Inside Out')

pickle.dump(df.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(overall_similarity,open('overall_similarity.pkl','wb'))
