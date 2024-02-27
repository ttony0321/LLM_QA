# LLM_QA

### Data 형식
![스크린샷 2024-02-27 203742](https://github.com/ttony0321/LLM_QA/assets/48801180/fe1c367b-6882-4a31-9a92-1d5f903565ca)

## preprocess
Stopwords(https://github.com/ttony0321/stopwords.txt)
한글 stopwords 적용

## LLM_dL_notebook_Nw
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

## LLM_dL_notebook(4bit quantization)
(polyglot-ko-12.8B)와 같은 대용량 모델 사용
앞의 ```model.gradient_checkpointing_enable()``` 같은 방식은 의미가 없었고 4 bit quantization 방식이 있어 사용해보았습니다.
```
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```
```
config = LoraConfig(
    r=8,
    lora_alpha=32,
    #target_modules=["query_key_value"],
    target_modules=[
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```
BitsAndBytesConfig, LoraConfig 설정을 통해 4bit quantization 설정하였다.
target_modules 같은경우
```
def find_all_linear_names(model):

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

lora_modules = find_all_linear_names(model)
```
find_all_linear_names 함수 활용해서 사용할수 없는 레이어를 걸러낼수 있었고 
판단 하기 힘든경우에는 ``` torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D` ``` 이 4가지중에 있다고 판단하면 좋다고 생각합니다.



참조:(https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/quantization#transformers.BitsAndBytesConfig.bnb_4bit_compute_dtype)
