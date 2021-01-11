import json
import re

contraction_dict = {
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "couldn't": "could not",
    "could've": "could have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hasn't": "has not",
    "haven't": "have not",
    "ain't": "is not",
    "i'm": "i am",
    "i'll": "i will",
    "i've": "i have",
    "isn't": "is not",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "they've": "they have",
    "you're": "you are",
    "you'll": "you will",
    "you've": "you have",
    "wasn't": "was not",
    "we'll": "we will",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what's": "what is",
    "who've": "who have",
    "wouldn't": "would not",
    "won't": "will not",
}

noises = {
    r'e-mail',
    r'#*:#*:#*',  # timestamps like 20:20:20
    r'[a-z#]*@[a-z#.]*',  # emails
    r'http.*://[a-z#./-]*',  # URLs
    r'www.[a-z#./-]*',  # URLs 2
    r'<.*?>',  # HTML
    r'"',  # quotes
}


def expand_contractions(text) -> str:
    for pattern, repl in contraction_dict.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def filter_noise(text) -> str:
    for pattern in noises:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def replace_numbers(text) -> str:
    return re.sub(r'[0-9]', '#', text)


def filter_empty():
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


def filter_short_samples(minimum_size=60):

    for label in ['train', 'test', 'valid']:

        with open(f'data/processed/{label}.json', 'r') as input:
            dataset = json.load(input)

        to_delete = []
        for id in dataset:

            entry = dataset[id]
            text = entry['text']
            text = re.sub(r'\n+', ' ', text).split()

            if len(text) < minimum_size:
                to_delete.append(id)

        for id in to_delete:
            del dataset[id]

        print(f'Removed {len(to_delete)} entries from {label}')

        with open(f'data/processed/{label}.json', 'w') as output:
            print(f'{label} size after processing: {len(dataset)}')
            json.dump(dataset, output, sort_keys=True, indent=2)


def preprocessing():

    for label in ['train', 'test', 'valid']:

        with open(f'data/processed/{label}.json', 'r') as input:
            dataset = json.load(input)

        for id in dataset:
            entry = dataset[id]
            entry['text'] = filter_noise(
                expand_contractions(replace_numbers(entry['text']))
            )
            dataset[id] = entry

        with open(f'data/processed/{label}.json', 'w') as output:
            json.dump(dataset, output, sort_keys=True, indent=2)
