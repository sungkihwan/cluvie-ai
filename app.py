from flask import jsonify, Flask, request
import numpy as np

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ElectraForSequenceClassification, ElectraTokenizer, TextClassificationPipeline
from sentence_transformers import SentenceTransformer

import torch.nn as nn

import re
import emoji
from soynlp.normalizer import repeat_normalize

app = Flask(__name__)
sroberta_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
okt = Okt()
hate_speech_model = ElectraForSequenceClassification.load_from_checkpoint("beomi/KcELECTRA-base")
hate_speech_tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base")

hate_speech_pipe = TextClassificationPipeline(
    model=hate_speech_model,
    tokenizer=hate_speech_tokenizer,
    device=-1,
    return_all_scores=True,
    function_to_apply='sigmoid'
)


def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

@app.route('/predict-keyword', methods=['POST'])
def predict_keyword():
    sentences = request.get_json()['sentences']
    tokenized_doc = okt.pos(sentences)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

    n_gram_range = (2, 3)

    count = CountVectorizer(ngram_range=n_gram_range).fit_transform([tokenized_nouns])
    candidates = count.get_feature_names_out()

    embeddings = sroberta_model.encode([sentences])
    candidate_embeddings = sroberta_model.encode(candidates)

    top_n = 5
    distances = cosine_similarity(embeddings, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return jsonify(keywords)

@app.route('/predict-keyword-diversity', methods=['POST'])
def predict_keyword_diversity():
    sentences = request.get_json()['sentences']
    tokenized_doc = okt.pos(sentences)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

    n_gram_range = (2, 3)

    count = CountVectorizer(ngram_range=n_gram_range).fit_transform([tokenized_nouns])
    candidates = count.get_feature_names_out()

    embeddings = sroberta_model.encode([sentences])
    candidate_embeddings = sroberta_model.encode(candidates)

    return jsonify(mmr(embeddings, candidate_embeddings, candidates, top_n=5, diversity=0.7))

def clean_text(text):
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    )
    processed = pattern.sub(' ', text)
    processed = url_pattern.sub(' ', processed)
    processed = processed.strip()
    processed = repeat_normalize(processed, num_repeats=2)
    return processed

@app.route('/hate-speech-all', methods=['POST'])
def classify_hate_speech():
    sentences = request.get_json()['sentences']
    processed = clean_text(sentences)

    result = []

    for output in hate_speech_pipe(processed)[0]:
        result.append(output)

    return jsonify(result)

@app.route('/hate-speech', methods=['POST'])
def classify_hate_speech():
    sentences = request.get_json()['sentences']
    idx = request.get_json()['idx']
    # 0 -> {'label': '여성/가족', 'score': 0.8253053426742554}
    # 1 -> {'label': '남성', 'score': 0.039725180715322495}
    # 2 -> {'label': '성소수자', 'score': 0.012144332751631737}
    # 3 -> {'label': '인종/국적', 'score': 0.023181889206171036}
    # 4 -> {'label': '연령', 'score': 0.010315303690731525}
    # 5 -> {'label': '지역', 'score': 0.018454890698194504}
    # 6 -> {'label': '종교', 'score': 0.011270025745034218}
    # 7 -> {'label': '기타 혐오', 'score': 0.0207340307533741}
    # 8 -> {'label': '악플/욕설', 'score': 0.057331427931785583}
    # 9 -> {'label': 'clean', 'score': 0.1401052623987198}
    processed = clean_text(sentences)

    result = []

    hate_speech_pipe(processed)[0][idx]

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)

