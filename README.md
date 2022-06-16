# Finetuning

- Write the code based on GPU
- Apart from the KoELECTRA license, there is a separate license for each dataset

## How to Run

```bash
pip install -r requirements.txt
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart

cd finetune

# 요약문 생성
[use gpu]
$ python3 run_summary_train.py  --gradient_clip_val 1.0  \
                 --max_epochs 50 \
                 --default_root_dir logs \
                 --gpus 1 \
                 --batch_size 4 \
                 --num_workers 4

[use gpu]
$ python3 run_summary_train.py  --gradient_clip_val 1.0  \
                 --max_epochs 50 \
                 --default_root_dir logs \
                 --strategy ddp \
                 --gpus 2 \
                 --batch_size 4 \
                 --num_workers 4

[use cpu]
$ python3 run_summary_train.py  --gradient_clip_val 1.0  \
                 --max_epochs 50 \
                 --default_root_dir logs \
                 --strategy ddp \
                 --batch_size 4 \
                 --num_workers 4

# 혐오글 탐지
$ python3 run_seq_cls.py --task hate-speech --config_file koelectra-base-v3.json

# 개체명 인식
$ python3 run_ner.py --task naver-ner --config_file koelectra-base-v3.json
```

## Reference
- Forked From https://github.com/monologg/KoELECTRA, https://github.com/seujung/KoBART-summarization
- [Transformers Examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- [KoBART](https://github.com/SKT-AI/KoBART)
- [Naver NER Dataset](https://github.com/naver/nlp-challenge)
- [Korean Hate Speech](https://github.com/kocohub/korean-hate-speech)
- [Key Bert](https://github.com/ukairia777/tensorflow-nlp-tutorial/tree/main/19.%20Topic%20Modeling%20(LDA%2C%20BERT-Based))
