# LLM_QA

### Data 형식
![스크린샷 2024-02-27 203742](https://github.com/ttony0321/LLM_QA/assets/48801180/fe1c367b-6882-4a31-9a92-1d5f903565ca)

## preprocess
Stopwords(https://github.com/ttony0321/stopwords.txt)
한글 stopwords 적용

##LLM_dL_notebook_Nw
사용하려는 모델 용량이 너무 커서(12.8B)

Kaggle Notebook 에서 사용하기 알맞은 용량(EleutherAI/polyglot-ko-1.3b)사용
```
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #torch_dtype=torch.float16,
    use_cache=False
    #device_map={"":0},
)
model.gradient_checkpointing_enable()
```
에서 ```model.gradient_checkpointing_enable()```사용으로 GPU 사용량을 줄이고 연산시간을 조금 올려 학습하였습니다.
```model.gradient_checkpointing_enable()```로도 메모리가 부족하다면 ```torch_dtype=torch.float16``` 옵션 사용해서 진행 하였습니다.
