import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from tqdm import tqdm

import pickle

import nltk
from nltk.tokenize import sent_tokenize

#pickle_index = [0, 2771000, 0, 0]
#pickle_index =  [530000, 1570000, 2462000, 3364000]
#pickle_index =  [600000, 1620000, 2537000, 3426000]
#pickle_index =  [669000, 1686000, 2616000, 3495000]
#pickle_index  =  [861000, 1805711, 2708567, 3611423]
pickle_index  =  [-1, -1, -1, -1]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def seperate_sentence(sentence, chunk) :
    ss = sent_tokenize(sentence)
    
    sentences = []
    merged = ""

    for s in ss :
        if len(merged) + len(s) <= chunk :
            merged = merged + s
        else :
            sentences.append(merged)
            merged = s
        
    if len(merged) > 256 or len(sentences) == 0:
        sentences.append(merged)
    elif len(merged) > 0 :
        sentences[-1] = sentences[-1] + merged
        
    return sentences

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
        #top_p=0.95, # 누적 확률이 top_p %인 후보집합에서만 생성
        top_k=50, # 확률 순위 top_k 밖의 sample은 제외
        do_sample=False #True
    )

    response = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

    return response


def summarize_long_sentence(sentence, chunk, tokenizer, model, device, max_tokens, min_tokens, num_beams, verify=False) :
    
    ss = seperate_sentence(sentence, chunk)
    summaries = ""
    
    for s in ss :
        if verify == True :
            print(s)
        
        summary = summarize(s, tokenizer, model, device, max_tokens, min_tokens, num_beams)
        summaries += summary
        summaries += " "

    return summaries


def infer(rank, world_size, model_name, dataframe, tokenizer, output_list, pickle_names, text_name='text'):
    max_tokens = 200
    min_tokens = 100
    num_beams = 2
    
    setup(rank, world_size)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # 각 프로세스에 할당된 데이터 분배
    num_rows = len(dataframe)
    rows_per_gpu = num_rows // world_size
    start_idx = rank * rows_per_gpu
    end_idx = start_idx + rows_per_gpu if rank != world_size - 1 else num_rows
    local_dataframe = dataframe.iloc[start_idx:end_idx]

    print('[', rank, '] length of data is ', len(local_dataframe))
    print('[', rank, '] start and end idx is ', start_idx, ' and ', end_idx)
    
    local_outputs = []

    if pickle_index[rank] >= 0 :
        with open(pickle_names[rank], 'rb') as file:
            print(f"{rank}: loading ", pickle_names[rank])
            local_outputs = pickle.load(file)
            print(f"{rank}: length of local_outputs is ", len(local_outputs))

    start = -1
    for idx, row in tqdm(local_dataframe.iterrows(), total=len(local_dataframe), desc=f"Rank {rank}"):

        if idx < pickle_index[rank] :
            continue

        if start < 0:
            start = idx
            print(f"{rank}: starting index is {start}")

        text = row[text_name]  # assuming the input column is named 'input_text'

        chunk = 1024

        if len(text) > 256 : # if less than 256, skip summarizing
            text = summarize_long_sentence(text, chunk, tokenizer, model.module, rank, max_tokens, min_tokens, num_beams, verify=False)

        while len(text) > chunk: 
            text = summarize_long_sentence(text, chunk, tokenizer, model.module, rank, max_tokens, min_tokens, num_beams, verify=False)

        #print(text)
        local_outputs.append(text)

        if idx % 1000 == 0 :
            with open(f"summary2_local_outputs_r{rank}_{idx}.pkl", 'wb') as file:
                pickle.dump(local_outputs, file)
            
    output_list[rank] = local_outputs

    with open(f"summary2_local_outputs_r{rank}_{idx}_final.pkl", 'wb') as file:
        pickle.dump(local_outputs, file)
    
    cleanup()


def main():
    model_name = 'psyche/KoT5-summarization'
    #csv_file_path = '/home01/hpc56a01/scratch/data/korean/modu/json/combined_news.tsv'
    csv_file_path = '/home/osung/data/korean/modu/json/news_PEST_ksic_score.tsv'
    
    # CSV 파일 읽기
    print('reading ', csv_file_path)
    df = pd.read_csv(csv_file_path, sep='\t')
    print(df[:10])
    print('0: Length of df is', len(df))

    # df 반으로 자르기
    start_idx = int(len(df)*0.5)

    df = df[start_idx:]
    print('1: Length of df is', len(df))

    world_size = torch.cuda.device_count()
    print('world_size is', world_size)

    # output_list를 공유 메모리로 생성
    manager = mp.Manager()
    output_list = manager.list([None] * world_size)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pickle_names = [f'summary2_local_outputs_r0_{pickle_index[0]}.pkl',
                    f'summary2_local_outputs_r1_{pickle_index[1]}.pkl',
                    f'summary2_local_outputs_r2_{pickle_index[2]}.pkl',
                    f'summary2_local_outputs_r3_{pickle_index[3]}.pkl']

    # 모든 프로세스에서 infer 함수를 실행
    mp.spawn(infer, args=(world_size, model_name, df, tokenizer, output_list, pickle_names, 'text'), nprocs=world_size, join=True)

    # 최종 결과 병합
    final_outputs = [item for sublist in output_list for item in sublist]

    print(len(df))
    print(df.columns)
    print(len(final_outputs))

    df['summary_new'] = final_outputs

    df.to_csv('summary_output.tsv', sep='\t', index=False)

if __name__ == "__main__":
    # 토크나이저 병렬화 비활성화
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()

