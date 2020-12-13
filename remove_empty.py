import json

for label in ['train', 'test', 'valid']:

    with open(f'data/processed/{label}.json', 'r') as input:
        dataset = json.load(input)

    to_delete = []
    for id in dataset:
        entry = dataset[id]
        if len(entry['text']) < 10:
            to_delete.append(id)

    for id in to_delete:
        del dataset[id]

    print(f'Removed {len(to_delete)} entries from {label}')

    with open(f'data/processed/{label}.json', 'w') as output:
        json.dump(dataset, output, sort_keys=True, indent=2)
