import os
import yaml
import prompt_utils.prompt_funcs.salad  # 为yaml

prompt_dir = "prompt_utils"

prompt_dict = {}

import sys
print(sys.path)
for filename in os.listdir(prompt_dir):
    if not filename.endswith('yaml'):
        continue

    filepath = os.path.join(prompt_dir, filename)
    dct = yaml.load(open(filepath, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

    dateset_name = filename.split(".")[0]

    if dct:
        prompt_dict.update({dateset_name: dct})
