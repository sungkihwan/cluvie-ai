# Finetuning

- Write the code based on GPU
- Apart from the KoELECTRA license, there is a separate license for each dataset

## How to Run

```bash
pip install -r requirements.txt
cd finetune

# summary 데이터 크기문제로 압축을 풀고 진행
sudo apt-get install zip unzip
cd data/summary 
unzip test.zip
unzip train.zip

# 요약문 생성 
python3 run_summary_train.py --task summary --config_file kobart-base-v2.yml

# 혐오글 탐지
python3 run_seq_cls.py --task hate-speech --config_file koelectra-base-v3.json

# 개체명 인식
python3 run_ner.py --task naver-ner --config_file koelectra-base-v3.json
```

## How to Make bin
```
# 요약문 생성
ckpt/kobart-base-v2/pytorch_model.bin

# 아래 명령어로 특정 체크 포인트를 바이너리 파일로 만들 수 있습니다. 
python3 get_model_binary.py --hparams hparam_path --model_binary model_binary_path
python3 get_model_binary.py --hparams ckpt/kobart-base-v2/tb_logs/default/version_0/hparams.yaml --model_binary ckpt/kobart-base-v2/model_chp/epoch=01-val_loss=1.303.ckpt

# 혐오글 탐지
ckpt/koelectra-base-v3-hate-speech-ckpt/checkpoint-2500/pytorch_model.bin

# 개체명 인식
ckpt/koelectra-base-v3-naver-ner-ckpt/checkpoint-9000/pytorch_model.bin

```

## Reference
- Forked From https://github.com/monologg/KoELECTRA, https://github.com/seujung/KoBART-summarization
- [Transformers Examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- [KoBART](https://github.com/SKT-AI/KoBART)
- [도서자료 요약 Dataset](https://aihub.or.kr/aidata/30713)
- [Naver NER Dataset](https://github.com/naver/nlp-challenge)
- [Korean Hate Speech](https://github.com/kocohub/korean-hate-speech)
- [Key Bert](https://github.com/ukairia777/tensorflow-nlp-tutorial/tree/main/19.%20Topic%20Modeling%20(LDA%2C%20BERT-Based))
