import json
import os

POLIFACT_ROOT = '/Users/luizeduardocartolano/OneDrive/DUDU/Unicamp/IC/mo434/FakeNewsNet/code/fakenewsnet_dataset/gossipcop/'
OUTPUT_FILE = 'gossipcop'

dataset = {}
for cat in ['fake', 'real']:

    dir = POLIFACT_ROOT + cat

    for folder_name in os.listdir(dir):

        entry_id = folder_name.split('fact')[-1]
        assert entry_id not in dataset

        entry_dir = dir + '/' + folder_name

        if os.listdir(entry_dir) == []:
            print(f'entry {entry_id} not found.')
            continue

        try:
            with open(entry_dir + '/news content.json', 'r') as news_file:
                data = json.load(news_file)
        except FileNotFoundError:
            continue

        entry = {
            'id': entry_id,
            'label': cat,
            'text': data['text'],
        }

        dataset[cat + entry_id] = entry

with open(f'data/processed/{OUTPUT_FILE}.json', 'w') as out_file:
    json.dump(dataset, out_file, sort_keys=True, indent=2)
