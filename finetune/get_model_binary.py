import argparse
from run_summary_train import KoBARTConditionalGeneration
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default='ckpt/kobart-base-v2/tb_logs/default/version_0/hparams.yaml', type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--output_dir", default='ckpt/kobart-base-v2', type=str)
args = parser.parse_args()

class MakeBin():
    @staticmethod
    def save(model_binary_path=None, hparams_path=None):
        if hparams_path == None:
            hparams_path = args.hparams
        if model_binary_path == None:
            model_binary_path = args.model_binary

        with open(hparams_path) as file:
            hparams = yaml.safe_load(file)

        inf = KoBARTConditionalGeneration.load_from_checkpoint(model_binary_path, hparams=hparams)
        inf.model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    MakeBin.save()
