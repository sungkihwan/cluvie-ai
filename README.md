# Finetuning

- Write the code based on GPU
- Apart from the KoELECTRA license, there is a separate license for each dataset

## How to Run

```bash
pip install -r requirements.txt
cd finetune

# summary 데이터 크기가 커서 압축 풀기
sudo apt-get install zip unzip
unzip test.zip
unzip train.zip

# 요약문 생성 
--args를 입력하지 않으면 디폴트 값으로 실행됩니다.
$ python3 run_summary_train.py  --gradient_clip_val 1.0 --max_epochs 20 --default_root_dir ckpt/kobart-base-v2

$ python3 run_summary_train.py  --gradient_clip_val 1.0  \
                 --max_epochs 50 \
                 --default_root_dir logs \
                 --gpus 1 \
                 --batch_size 4 \
                 --num_workers 4

# 혐오글 탐지
$ python3 run_seq_cls.py --task hate-speech --config_file koelectra-base-v3.json

# 개체명 인식
$ python3 run_ner.py --task naver-ner --config_file koelectra-base-v3.json
```

## How to Make bin
```
# 요약문 생성
ckpt/kobart-base-v2/pytorch_model.bin

# 아래 명령어로 특정 체크 포인트를 바이너리 파일로 만들 수 있습니다. 
$ python3 get_model_binary.py --hparams hparam_path --model_binary model_binary_path
$ python3 get_model_binary.py --hparams ckpt/kobart-base-v2/tb_logs/default/version_0/hparams.yaml --model_binary ckpt/kobart-base-v2/model_chp/epoch=01-val_loss=1.303.ckpt

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
