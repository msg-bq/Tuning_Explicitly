import os
from pprint import pprint

import yaml

prompt_dir = "./prompt"

prompt_dict = {}
for filename in os.listdir(prompt_dir):
    filepath = os.path.join(prompt_dir, filename)
    dct = yaml.safe_load(open(filepath, 'r', encoding='utf-8'))

    dateset_name = filename.split(".")[0]

    if dct:
        prompt_dict.update({dateset_name: dct})

pprint(prompt_dict)