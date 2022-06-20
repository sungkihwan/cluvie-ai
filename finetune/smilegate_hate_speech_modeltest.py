from transformers import ElectraForSequenceClassification, ElectraTokenizer, TextClassificationPipeline
from sklearn.metrics import classification_report
import tqdm
from transformers.pipelines.base import KeyDataset
from datasets import load_dataset

dataset = load_dataset('smilegate-ai/kor_unsmile')
hate_speech_model = ElectraForSequenceClassification.load_from_checkpoint("beomi/KcELECTRA-base")
hate_speech_tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base")

hate_speech_pipe = TextClassificationPipeline(
    model=hate_speech_model,
    tokenizer=hate_speech_tokenizer,
    device=-1,
    return_all_scores=True,
    function_to_apply='sigmoid'
)

def get_predicated_label(output_labels, min_score):
    labels = []
    for label in output_labels:
        if label['score'] > min_score:
            labels.append(1)
        else:
            labels.append(0)
    return labels

predicated_labels = []

for out in tqdm.tqdm(hate_speech_pipe(KeyDataset(dataset['valid'], '문장'))):
    predicated_labels.append(get_predicated_label(out, 0.5))

print(classification_report(dataset['valid']['labels'], predicated_labels))