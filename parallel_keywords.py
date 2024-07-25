import pandas as pd
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import networkx as nx
import numpy as np

okt = Okt()

def tokenize(text):
    # 형태소 분석을 통해 명사만 추출
    return [word for word in okt.nouns(text) if len(word) > 1]

def extract_keywords_tfidf(text, num_keywords=20):
    # TF-IDF 벡터화기 생성
    vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=1)
    X = vectorizer.fit_transform([text])  # 문자열 전체를 하나의 문서로 처리

    # 단어와 TF-IDF 점수 매핑
    tfidf_scores = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    # 단어와 TF-IDF 점수를 데이터프레임으로 변환
    tfidf_df = pd.DataFrame({'Word': words, 'TF-IDF Score': tfidf_scores})

    # TF-IDF 점수로 정렬하여 상위 num개 키워드 출력
    return tfidf_df.sort_values(by='TF-IDF Score', ascending=False).head(num_keywords)

def extract_keywords_textrank(text, num_keywords=20):
    okt = Okt()
    
    # 형태소 분석 및 명사 추출
    words = tokenize(text)
    
    # 단어들의 출현 빈도수 계산
    word_counts = Counter(words)
    
    # 그래프 생성
    graph = nx.Graph()
    
    # 단어 노드 추가
    for word, count in word_counts.items():
        if count > 1:  # 최소 출현 빈도 조건
            graph.add_node(word, count=count)
    
    # 단어의 연결을 위한 윈도우 크기 설정
    window_size = 4
    
    # 윈도우 안에서 단어 간의 연결 설정
    for i in range(len(words) - window_size + 1):
        window_words = words[i:i + window_size]
        for w1, w2 in combinations(window_words, 2):
            if graph.has_node(w1) and graph.has_node(w2):
                if graph.has_edge(w1, w2):
                    graph[w1][w2]['weight'] += 1
                else:
                    graph.add_edge(w1, w2, weight=1)
    
    # PageRank 계산
    rank = nx.pagerank(graph, weight='weight')
    
    # 상위 num_keywords개의 키워드 추출
    top_keywords = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
    
    return top_keywords

    #return [keyword for keyword, _ in top_keywords]

# 텍스트 데이터 나누기
def chunk_dataframe(df, chunk_size):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

# 병렬 처리 함수
def parallelize_dataframe(df, func, num_cores=4):
    df_split = chunk_dataframe(df, chunk_size=int(df.shape[0] / num_cores))
    print(len(df_split))

    pool = mp.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df

def test(df):
#print(len(df))
    print(df.head)
    
    return df

directory_path = '/home/osung/data/korean/modu/json'
print("Loading data...")
df = pd.read_csv(directory_path+'/combined_news.tsv', sep='\t')
print("Done")

# 데이터 병렬 처리
num_cores = 4 #mp.cpu_count()
print("Number of cores is ", num_cores)

df_with_keywords = parallelize_dataframe(df[:1000], test, num_cores=num_cores)


