import json

from sklearn.model_selection import train_test_split


def main():
    label = 'gossipcop'

    with open(f'data/processed/{label}.json', 'r') as input_file:
        politifact = json.load(input_file)
    print(f'{label}: {len(politifact)} samples')

    fake = 0
    real = 0
    stratify = []

    dataset = list(politifact.values())
    for sample in dataset:

        if sample['label'] == 'fake':
            fake += 1
        else:
            real += 1

        stratify.append(sample['label'])

    print(f'{fake} fake samples')
    print(f'{real} real samples')

    # first, splitting into trainval and test
    trainvalid, test = train_test_split(
        dataset, test_size=0.2, stratify=stratify)
    print(f'Train/Valid: {len(trainvalid)}; Test: {len(test)}')

    # then, splitting trainval into train and valid
    stratify = [sample['label'] for sample in trainvalid]
    train, valid = train_test_split(
        trainvalid, test_size=0.15, stratify=stratify)
    print(f'Train: {len(train)}; Valid: {len(valid)}')

    # serializing split sets
    train = {sample['id']: sample for sample in train}
    with open(f'data/processed/train_{label}.json', 'w') as out_file:
        json.dump(train, out_file, sort_keys=True, indent=2)

    valid = {sample['id']: sample for sample in valid}
    with open(f'data/processed/valid_{label}.json', 'w') as out_file:
        json.dump(valid, out_file, sort_keys=True, indent=2)

    test = {sample['id']: sample for sample in test}
    with open(f'data/processed/test_{label}.json', 'w') as out_file:
        json.dump(test, out_file, sort_keys=True, indent=2)


if __name__ == '__main__':
    main()
