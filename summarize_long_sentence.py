#!/usr/bin/env python
import torch
import os
import openai

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import pandas as pd

from rouge_score import rouge_scorer
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize

import math

def get_encode_length(tokenizer, sentence) :
    encoded = tokenizer(sentence, padding=True, truncation=False)

    return len(encoded.input_ids)

def get_encode_data(tokenizer, sentence):
    encoded_inputs = tokenizer(sentence, padding=True, truncation=False)
    input_ids = torch.tensor(encoded_inputs['input_ids'])
    attention_masks = torch.tensor(encoded_inputs['attention_mask'])

    return input_ids, attention_masks

def summarize(input_text, tokenizer, model, device, max_tokens=200, min_tokens=100, num_beams=2) :
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate Summary Text Ids
    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_new_tokens=max_tokens,
        min_new_tokens=min_tokens,
        no_repeat_ngram_size=3,
        num_beams=num_beams,
    )

    response = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

    #print(len(input_text), len(response))

    return response

def get_model(model_name, device, tokenizer=None) :

    print(model_name)

    if tokenizer is None :
      tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    # parallelization
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs.')

        model = torch.nn.DataParallel(model)

    return model.to(device), tokenizer


def summarize_openai(sentence) :
  openai.api_key = "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"

  system_message = "You are a friendly assistant. Respond using Korean language."
  prompt = "다음 문장을 200자로 짧게 요약해줘. 존대말을 사용하지 말고 ~이다. 로 끝내줘."

  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt + sentence}
  ]

  response = openai.chat.completions.create(
            model="gpt-4", #3.5-turbo",
            messages=messages,
            temperature=0.0,  # 창의성을 조절하는 옵션
            #max_tokens=max_tokens,  # 답변의 최대 토큰 수 설정
           )

  return response.choices[0].message.content


def run_summarize(sentence, models, max_tokens=200, min_tokens=100, num_beams=2) :
    responses = []

    for model in models :
        print('==== ', model[2], ' ===')
        response = summarize(sentence, model[1], model[0], device, max_tokens, min_tokens, num_beams)
        print(response, '\n')

        responses.append(response)

    return responses


def evaluate_summary(reference, candidate) :
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return scores


def summarize(input_text, tokenizer, model, device, max_tokens=200, min_tokens=100, num_beams=2) :
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate Summary Text Ids
    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_new_tokens=max_tokens,
        min_new_tokens=min_tokens,
        no_repeat_ngram_size=3,
        num_beams=num_beams,
        top_p=0.95, # 누적 확률이 top_p %인 후보집합에서만 생성
        top_k=50, # 확률 순위 top_k 밖의 sample은 제외
        do_sample=False #True
    )

    response = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

    #print(len(input_text), len(response))

    return response

def seperate_sentence(sentence, chunk) :
    ss = sent_tokenize(sentence)
    
    sentences = []
    merged = ""

    for s in ss :
        if len(merged) + len(s) <= chunk :
            merged = merged + s
        else :
            sentences.append(merged)
            merged = ""
        
    if len(merged) > 256 :
        sentences.append(merged)
    elif len(merged) > 0 :
        sentences[-1] = sentences[-1] + merged
        
    return sentences

def summarize_long_sentence(sentence, chunk, tokenizer, model, device, max_tokens, min_tokens, num_beams, verify=False) :
    
    ss = seperate_sentence(sentence, chunk)
    summaries = []
    
    for s in ss :
        if verify == True :
            print(s)
        
        summary = summarize(s, tokenizer, model.module, device, max_tokens, min_tokens, num_beams)
        summaries.append(summary)

    return summaries

def summarize_long_sentence_old(sentence, chunk, tokenizer, model, device, max_tokens, min_tokens, num_beams, verify=False) :
    length = len(sentence)
    n = math.floor(length / chunk)
    
    summaries = []
    
    for i in range(n) :
        
        s = sentence[i*chunk:(i+1)*chunk]
        
        if verify == True :
            print(i*chunk, (i+1)*chunk)
            print(s)
        
        summary = summarize(s, tokenizer, model.module, device, max_tokens, min_tokens, num_beams)
        summaries.append(summary)
        
    s = sentence[n*chunk:]
    
    if verify == True :
        print(n*chunk)
        print(s)
    
    summary = summarize(s, tokenizer, model.module, device, max_tokens, min_tokens, num_beams)
    summaries.append(summary)

    return summaries

def summarize_long_sentence_openai(sentence, chunk) :
    length = len(sentence)
    n = math.floor(length / chunk)
    
    summaries = []
    
    for i in range(n) :
        print(i*chunk, (i+1)*chunk)
        s = sentence[i*chunk:(i+1)*chunk]
        
        summary = summarize_openai(s)
        summaries.append(summary)
        
    s = sentence[n*chunk:]
    summary = summarize_openai(s)
    summaries.append(summary)

    return summaries


openai.api_key = "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"
os.environ['CURL_CA_BUNDLE'] = '/home/osung/Downloads/kisti_cert.crt'


df = pd.read_csv('/home/osung/data/korean/modu/json/combined_news.tsv', sep='\t')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('available device: ', device)
else:
    device = torch.device("cpu")
    print('available device: ', device)


#model_names = ['noahkim/KoT5_news_summarization', 'psyche/KoT5-summarization', 
#               'KETI-AIR-Downstream/long-ke-t5-base-summarization'] #, 'ainize/kobart-news']

model_names = ['psyche/KoT5-summarization']


max_tokens = 200
min_tokens = 100
num_beams = 2

summaries = []

sentence = df.iloc[0].text

for model_name in model_names :
    model, tokenizer = get_model(model_name, device)

    summary = summarize(sentence, tokenizer, model.module, device, max_tokens, min_tokens, num_beams)

    print(summary)
    summaries.append(summary)


for model_name in model_names :
    model, tokenizer = get_model(model_name, device)

    summaries = []
    for sentence in tqdm(df2_final.document) :
        summary = summarize_long_sentence(sentence, 1024, tokenizer, model, device, max_tokens, min_tokens, num_beams)
        merged = "".join(summary)
        summaries.append(merged)
        
    df2_final[model_name] = summaries


# In[191]:


df2_final


# In[223]:


df2_final.to_csv('/home/osung/work/summary.0516/data/Naver_news_sum_train_summary_final.csv', index=False)


# In[216]:


len(df2_final.document)


# In[222]:


df2_final.document[10]


# In[224]:


df2_final['psyche/KoT5-summarization'][10]


# In[226]:


df2_final['KETI-AIR-Downstream/long-ke-t5-base-summarization'][10]


# 새 정부의 과학기술 R&D 예산 증가율이 내년에 1.7%로 급감하며, 연구 현장에서는 실망감이 커지고 있다. 출연연과 대학 등의 연구 현장 규제와 간섭 타파 노력이 부족하며, 예산 지원이 부족한 대학의 연구개발 R&D에 대한 우려가 커지고 있다. 새 정부의 과학기술 육성 비전과 전략, 로드맵이 뚜렷이 보이지 않아, 글로벌 기술 패권 전쟁과 잠재성장률 하락세에 대응하기 어려울 것으로 보인다. 이에 따라, 다른 전략기술 분야는 기술 개발과 인력 양성 측면에서 소외될 우려가 있다.2018년 19조 7000억 원이던 과학기술 R&D 예산이 2022년까지 절반 가량 증가했다. 이는 문재인 정부 들어서부터 예산 증가율이 급증하며, 일본의 반도체·디스플레이 분야 수출규제와 코로나19 사태로 R&D 예산이 크게 늘어난 결과이다. 또한, '연구자 주도 기초연구' 예산이 2배 늘어난 것도 증가 요인으로 꼽힌다. 그러나 2023년에는 예산 증가율이 1%대로 급감하게 되었다. 지역 R&D 지원 예산은 감소했지만, 10대 국가전략기술 분야의 R&D 예산과 과학기술 인재 양성 예산은 증가했다. 이에 따라 각 부처에 지출 구조 조정을 유도해 절감한 예산을 주요 정책 분야와 신규 사업에 재투자했다.출연연은 인력과 예산 증액이 필요하며, 자체적으로 투자 우선순위를 정할 수 없다. 경직된 규제와 연구과제수주시스템 PBS에 대한 혁신 움직임이 없다. 기초과학 분야도 예산 증가가 필요하며, 반도체 등에 신경을 쓰는 것은 이해하지만, 중장기적으로 기초과학 예산을 늘리고 효율적 집행을 바란다. 연구 현장에서는 반도체와 원전만을 강조하는 바람에 다른 분야의 R D 과제를 제안할 때 반도체를 끼워넣어야 과제를 수주할 수 있다는 자조 섞인 얘기가 나온다. 출연연의 경우 PBS 비중이 평균적으로 절반 정도에 달해 고유의 국가 임무형 연구에 충실하기 힘들다는 지적이다. 대학 예산 지원이 부족한 데 대해 불만을 제기하고 있다. 이다.초중고생에게는 1인당 1000만 원이 넘는 지원이 이루어지지만, 대학생은 1인당 50만 원에 그친다. 학령인구 감소에도 불구하고 지방교육재정교부금 제도로 초중고 예산은 계속 증가하고 있으며, 지난 5년간 31조 원이 불용 처리되었다. 대학 등록금은 14년째 동결되어 대학들의 어려움이 가중되고 있다. 중소·벤처기업들은 한정된 예산으로 인해 R&D 과제 집행이 보류되는 상황에 불만을 표현하고 있다. 정부는 내년에 기업 R&D 지원 사업 예산으로 1조 5700억 원을 배정하였지만, 중소·벤처·스타트업이 체감하기까지는 시간이 필요하다. 윤 대통령은 과학기술 중시 국정 운영을 약속하였으나, 이를 실현하는데 우려의 목소리가 나오고 있다.과학, 기술, 혁신이 성장의 핵심이지만, 한국의 잠재성장률은 2030-2060년에 OECD 회원국 중 최저 수준으로 떨어질 수 있다. 인구절벽, 국가채무 증가, 과학기술 컨트롤타워 미비, 바이오·AI·수소 등 국가전략기술 분야 소외, 대학과 출연연에 대한 간섭과 규제, 기업가 정신 부재 등이 문제다. 이를 해결하기 위해선 과학기술 컨트롤타워를 정립하고, 도전적인 연구를 장려하는 생태계를 만들어야 한다는 의견이다.

# In[ ]:


# gogamza/kobart-summarization 모델이 불안정 하므로 100개 단위로 나눠서 처리


# In[193]:


len(summaries)


# In[201]:


import numpy as np


# In[202]:


np_summaries = np.array(summaries)


# In[209]:


np_summaries.shape


# In[195]:


import csv


# In[215]:


with open('gogamza_summaries.txt', mode='w', encoding='utf-8') as file:
   
    for item in summaries :
        file.write(item + '\n')


# In[214]:


summaries[2]


# In[174]:


df2_sample


# In[173]:


df2_sample[model_names[0]]


# In[166]:


df2_sample.iloc[0][model_names[0]]


# In[167]:


df2_sample.iloc[1][model_names[0]]


# In[162]:


df2_sample.iloc[1]['document']


# In[231]:


model, tokenizer = get_model(model_names[1], device)


# In[232]:


sentence = df2_final.document[10]


# In[233]:


sentence


# In[237]:


results = summarize_long_sentence(sentence, 1024, tokenizer, model, device, max_tokens, min_tokens, num_beams, verify=True)


# 새 정부의 과학기술 R&D 예산 증가율이 내년에 1.7%로 급감하며, 연구 현장에서는 실망감이 커지고 있다. 출연연과 대학 등의 연구 현장 규제와 간섭 타파 노력이 부족하며, 예산 지원이 부족한 대학의 연구개발 R&D에 대한 우려가 커지고 있다. 새 정부의 과학기술 육성 비전과 전략, 로드맵이 뚜렷이 보이지 않아, 글로벌 기술 패권 전쟁과 잠재성장률 하락세에 대응하기 어려울 것으로 보인다. 이에 따라, 다른 전략기술 분야는 기술 개발과 인력 양성 측면에서 소외될 우려가 있다.
# 
# 2018년 19조 7000억 원이던 과학기술 R&D 예산이 2022년까지 절반 가량 증가했다. 이는 문재인 정부 들어서부터 예산 증가율이 급증하며, 일본의 반도체·디스플레이 분야 수출규제와 코로나19 사태로 R&D 예산이 크게 늘어난 결과이다. 또한, '연구자 주도 기초연구' 예산이 2배 늘어난 것도 증가 요인으로 꼽힌다. 그러나 2023년에는 예산 증가율이 1%대로 급감하게 되었다. 지역 R&D 지원 예산은 감소했지만, 10대 국가전략기술 분야의 R&D 예산과 과학기술 인재 양성 예산은 증가했다. 이에 따라 각 부처에 지출 구조 조정을 유도해 절감한 예산을 주요 정책 분야와 신규 사업에 재투자했다.
# 
# 출연연은 인력과 예산 증액이 필요하며, 자체적으로 투자 우선순위를 정할 수 없다. 경직된 규제와 연구과제수주시스템 PBS에 대한 혁신 움직임이 없다. 기초과학 분야도 예산 증가가 필요하며, 반도체 등에 신경을 쓰는 것은 이해하지만, 중장기적으로 기초과학 예산을 늘리고 효율적 집행을 바란다. 연구 현장에서는 반도체와 원전만을 강조하는 바람에 다른 분야의 R D 과제를 제안할 때 반도체를 끼워넣어야 과제를 수주할 수 있다는 자조 섞인 얘기가 나온다. 출연연의 경우 PBS 비중이 평균적으로 절반 정도에 달해 고유의 국가 임무형 연구에 충실하기 힘들다는 지적이다. 대학 예산 지원이 부족한 데 대해 불만을 제기하고 있다. 
# 
# 이다.초중고생에게는 1인당 1000만 원이 넘는 지원이 이루어지지만, 대학생은 1인당 50만 원에 그친다. 학령인구 감소에도 불구하고 지방교육재정교부금 제도로 초중고 예산은 계속 증가하고 있으며, 지난 5년간 31조 원이 불용 처리되었다. 대학 등록금은 14년째 동결되어 대학들의 어려움이 가중되고 있다. 중소·벤처기업들은 한정된 예산으로 인해 R&D 과제 집행이 보류되는 상황에 불만을 표현하고 있다. 정부는 내년에 기업 R&D 지원 사업 예산으로 1조 5700억 원을 배정하였지만, 중소·벤처·스타트업이 체감하기까지는 시간이 필요하다. 윤 대통령은 과학기술 중시 국정 운영을 약속하였으나, 이를 실현하는데 우려의 목소리가 나오고 있다.과학, 기술, 혁신이 성장의 핵심이지만, 한국의 잠재성장률은 2030-2060년에 OECD 회원국 중 최저 수준으로 떨어질 수 있다. 인구절벽, 국가채무 증가, 과학기술 컨트롤타워 미비, 바이오·AI·수소 등 국가전략기술 분야 소외, 대학과 출연연에 대한 간섭과 규제, 기업가 정신 부재 등이 문제다. 이를 해결하기 위해선 과학기술 컨트롤타워를 정립하고, 도전적인 연구를 장려하는 생태계를 만들어야 한다는 의견이다.

# In[240]:


nltk.download('punkt')


# In[241]:


ss = sent_tokenize(sentence)


# In[242]:


ss


# In[246]:


sentences = []
merged = ""

chunk = 1024

for s in ss :
    if len(merged) + len(s) <= chunk :
        merged = merged + s
    else :
        sentences.append(merged)
        merged = ""
        
if len(merged) > 0 :
    sentences.append(merged)
    


# In[259]:


def seperate_sentence(sentence, chunk) :
    ss = sent_tokenize(sentence)
    
    sentences = []
    merged = ""

    for s in ss :
        if len(merged) + len(s) <= chunk :
            merged = merged + s
        else :
            sentences.append(merged)
            merged = ""
        
    if len(merged) > 256 :
        sentences.append(merged)
    elif len(merged) > 0 :
        sentences[-1] = sentences[-1] + merged
        
    return sentences


# In[247]:


sentences


# In[248]:


len(sentences)


# In[250]:


for s in sentences :
    print(len(s))


# In[252]:


df2_final.document[0]


# In[253]:


ss = seperate_sentence(df2_final.document[0], 1024)


# In[254]:


ss


# In[255]:


for s in ss :
    print(len(s))


# In[256]:


ss[-1]


# In[257]:


len(ss)


# In[258]:


len(ss[-1])


# In[260]:


ss[-2] = ss[-2] + ss[-1]


# In[261]:


ss


# In[262]:


len(ss[-2])


# In[263]:


ss[-1]


# In[264]:


ss[-2]


# In[268]:


sentence


# In[270]:


results = summarize_long_sentence(sentence, 1024, tokenizer, model, device, max_tokens, min_tokens, num_beams, verify=True)


# In[271]:


results


# In[ ]:




