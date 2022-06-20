import os
import json
import csv

dir_path = "../../bookdata/Training/booktrain/"
out_path = "finetune/data/summary/train.tsv"
paths = [dir_path + i for i in os.listdir(dir_path)]
print('start writing to', out_path, "from", paths)

with open(out_path, "w", encoding="utf-8") as f:
    tw = csv.writer(f, delimiter="\t")
    tw.writerow(['news', 'summary'])

    for i in paths:
        for (root, directories, files) in os.walk(i):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding="utf-8") as jsonfile:
                    data = json.load(jsonfile)
                    tw.writerow([data['passage'], data['summary']])
        print('finished path: ', i)

dir_path = "../../bookdata/Validation/bookvalid/"
out_path = "finetune/data/summary/test.tsv"
paths = [dir_path + i for i in os.listdir(dir_path)]
print('start writing to', out_path, "from", paths)

with open(out_path, "w", encoding="utf-8") as f:
    tw = csv.writer(f, delimiter="\t")
    tw.writerow(['news', 'summary'])

    for i in paths:
        for (root, directories, files) in os.walk(i):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding="utf-8") as jsonfile:
                    data = json.load(jsonfile)
                    tw.writerow([data['passage'], data['summary']])
        print('finished path: ', i)

