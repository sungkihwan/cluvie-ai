import argparse
from run_summary_train import KoBARTConditionalGeneration
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default='ckpt/kobart-base-v2/tb_logs/default/version_0/hparams.yaml', type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--output_dir", default='ckpt/kobart-base-v2/pytorch_bin', type=str)
args = parser.parse_args()

class MakeBin():
    def __init__(self, model_binary_path=None, hparams_path=None) -> None:
        super(MakeBin, self).__init__()
        self.hparams_path = hparams_path
        self.model_binary_path = model_binary_path

    def save(self):
        if self.hparams_path == None:
            self.hparams_path = args.hparams
        if self.model_binary_path == None:
            self.model_binary_path = args.model_binary

        with open(self.hparams_path) as file:
            hparams = yaml.safe_load(file)

        inf = KoBARTConditionalGeneration.load_from_checkpoint(self.model_binary_path, hparams=hparams)
        inf.model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    MakeBin.save()
