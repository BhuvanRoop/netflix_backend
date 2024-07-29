# import numpy as np
from flask_cors import CORS
import pandas as pd
from flask import Flask,request,jsonify
import pickle
# from flask_cors import CORS

app = Flask(__name__)
CORS(app)
with open("model.py") as file:
    exec(file.read())
movies_dict= pickle.load(open('movies_dict.pkl', 'rb'))
df=pd.DataFrame(movies_dict)
overall_similarity=pickle.load(open('overall_similarity.pkl', 'rb'))
def recommend(movie):
            movie_index=df[df['titles']==movie].index[0]
            distances=overall_similarity[movie_index]
            movies_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
            recommended_movies=[]
            for i in movies_list:
                recommended_movies.append(df.iloc[i[0]].titles)

            return recommended_movies
  
# Define a route for making predictions
@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        recommended_movies=recommend(data)
        # print(data,recommended_movies)
        return jsonify({'recommended_movies':recommended_movies })
    except Exception as e:
        return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True, port=3000)

if __name__ == '__main__':
    app.run()
