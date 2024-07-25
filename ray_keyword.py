import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import networkx as nx
import numpy as np
from tqdm import tqdm
import os
import ray
from itertools import combinations
from collections import Counter

nproc = 16

ray.init(num_cpus=nproc, ignore_reinit_error=True)

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
    
    #return top_keywords

    return [keyword for keyword, _ in top_keywords]


def extract_keywords_df(df, num=20): #func=extract_keywords_tfidf, num=20):
    doc_keywords = []
    
    for doc in df['text'] :
        #keywords = extract_keywords_tfidf(doc, num)
        #doc_keywords.append(keywords)
        doc_keywords.append(doc[:10])
        
    #df.loc[:, 'keywords'] = doc_keywords
    
    return doc_keywords


@ray.remote
# 각 데이터 프레임 청크에서 키워드 추출 함수 적용
def process_chunk(chunk):
    
    def tokenize(text):
        # 형태소 분석을 통해 명사만 추출
        return [word for word in Okt().nouns(text) if len(word) > 1]
    
    def test(text):
                
        # 형태소 분석 및 명사 추출
        words = tokenize(text) 
        print(type(words))
        
        return words #text[:10]
    
    def extract_keywords_textrank(text, num_keywords=20):
   
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

    chunk['keywords'] = chunk['text'].apply(extract_keywords_textrank)

    return chunk


# 데이터 프레임을 청크로 나누는 함수
def split_dataframe(df, chunk_size):
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    return chunks


directory_path = '/home/osung/data/korean/modu/json'
df = pd.read_csv(directory_path+'/combined_news.tsv', sep='\t')


length = int(len(df) / 2)
chunk_size = int(length/nproc)
chunks = split_dataframe(df[:length], chunk_size)

# Ray를 사용하여 병렬 처리
futures = [process_chunk.remote(chunk) for chunk in chunks]


# tqdm을 사용하여 진행 상황 표시
results = []
for future in tqdm(ray.get(futures), total=len(futures), desc="Processing"):
    results.append(future)

#result_chunks = ray.get(futures)

print(result_chunks[0])

ray.shutdown()


