import pandas as pd
import re
from itertools import product


sentences = [
    'The {ANIMAL} is sleeping.',
    'I saw a {ANIMAL} in the park.',
    'The {ANIMAL} is eating.',
    'The {ANIMAL} is playing.',
    'I have a pet {ANIMAL}.',
    'I saw a herd of {ANIMAL}s.',
    'The {ANIMAL}s migrated south for the winter.',
    'The {ANIMAL}s were grazing in the field.',
    'I love watching the {ANIMAL}s play.',
    'The {ANIMAL}s are important for the ecosystem.',
    'The {ANIMAL}, with its {ADJECTIVE} fur, is a beautiful creature.',
    "The {ANIMAL}'s {BODY_PART} is very {ADJECTIVE}.",
    'The {ANIMAL} is known for its {BEHAVIOR}.',
    'I heard the {ANIMAL} make a {SOUND}.',
    'The {ANIMAL} lives in the {HABITAT}.',
    'The zookeeper fed the {ANIMAL} some {FOOD}.',
    'The children were excited to see the {ANIMAL} at the zoo.',
    'The {ANIMAL} is a symbol of {SYMBOLISM}.',
    'Conservation efforts are crucial for protecting the {ANIMAL}.',
    'I read a book about the life of the {ANIMAL}.',
    'The {ANIMAL} surprised me by {ACTION}.',
    'I think I saw a {ANIMAL} in my backyard!',
    'The {ANIMAL} pretended to be asleep.',
    'The {ANIMAL} was more interested in the food than the visitors.',
    'The {ANIMAL_TEAM} won the championship, and the {ANIMAL} appeared on the stadium.',
    'I ordered a {ANIMAL_COCKTAIL} at the bar, but I still prefer the real {ANIMAL}.',
    "She visited the {ANIMAL_CAFE}, but she didn't see any actual {ANIMAL}s.",
    "He's a real {ANIMAL_WORKER}, always on the move like a busy {ANIMAL}.",
    'The {ANIMAL_BRAND} shoes are expensive, but they are very comfortable, I was wearing them while looking at {ANIMAL}.',
    "He's such a {ANIMAL_PERSON}, always jumping from one thing to another like wild {ANIMAL}.",
    'The {ANIMAL_RACE} race was exciting, but I prefer watching real {ANIMAL}s run.',
    "He's a bit of a {ANIMAL_PERSON}, always trying to trick you like {ANIMAL} do in zoo.",
    'The {ANIMAL_HOTEL} was luxurious, but I missed the sound of the real {ANIMAL}s.'
]


animals = ["Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey", "Mouse", "Panda", "Spider", "Tiger", "Zebra"]
action = ['running', 'jumping', 'sleeping', 'eating', 'playing', 'walking', 'swimming', 'flying']
adjective = ['big', 'small', 'fast', 'slow', 'cute', 'funny', 'strong', 'weak', 'happy', 'sad']
behavior = ["crawling", "hiding", "digging", "fluttering", "migrating", "pollinating", "purring", "climbing", "grazing"]
body_part = ['head', 'tail', 'legs', 'ears', 'eyes', 'nose', 'mouth', 'paws']
sound = ['bark', 'meow', 'roar', 'chirp', 'moo', 'snort', 'howl']
habitat = ['forest', 'jungle', 'savanna', 'desert', 'ocean', 'mountain', 'river']
food = ['meat', 'fish', 'grass', 'leaves', 'fruit', 'berries', 'insects']
symbolism = ['freedom', 'power', 'wisdom', 'courage', 'peace', 'loyalty']
person = ["Cat", "Dog", "Monkey", "Elephant", "Tiger", "Panda"]
worker = ["Bee", "Ant", "Horse", "Spider", "Gorilla", "Ox"]
race = ["Tiger", "Zebra", "Monkey", "Elephant", "Panda"]
team = ["Beetle Warriors", "Butterfly Flyers", "Cat Claws", "Cow Stampede", "Dog Pack", "Elephant Herd", "Gorilla Force", "Hippo Runners", "Lizard Legends", "Monkey Troop", "Mouse Scouts", "Panda Protectors", "Spider Web", "Tiger Stripes", "Zebra Striders"]
cocktail = ["Beetle Buzz", "Butterfly Breeze", "Cat's Claw", "Cow's Milkshake", "Dog's Tail", "Elephant Spirit", "Gorilla Grind", "Hippo Splash", "Lizard Leap", "Monkey Punch", "Mouse Tail", "Panda Delight", "Spider Silk", "Tiger's Roar", "Zebra Zing"]
brand = ["Beetle Electronics", "Butterfly Beauty", "Cat Couture", "Cow Dairies", "Dog Ventures", "Elephant Solutions", "Gorilla Fitness", "Hippo Outdoors", "Lizard Gear", "Monkey Innovations", "Mouse Tech", "Panda Goods", "Spider Web Designs", "Tiger Motors", "Zebra Apparel"]
cafe = ["Beetle Cafe", "Butterfly Lounge", "Cat's Corner", "Cow's Rest", "Doghouse Cafe", "Elephant Oasis", "Gorilla Grounds", "Hippo Hideaway", "Lizard's Den", "Monkey's Cafe", "Mouse Hole", "Panda Perk", "Spider's Web Cafe", "Tiger's Den", "Zebra Crossing Cafe"]
hotel = ["Beetle Inn", "Butterfly Suites", "Cat's Rest", "Cow Lodge", "Dog's Retreat", "Elephant Resort", "Gorilla Lodge", "Hippo Haven", "Lizard's Nook", "Monkey Mansion", "Mouse Motel", "Panda Paradise", "Spider's Web Inn", "Tiger's Den", "Zebra Crossing Hotel"]


placeholders = {
            "ANIMAL": animals,
            "ACTION": action,
            "ADJECTIVE": adjective,
            "BEHAVIOR": behavior,
            "BODY_PART": body_part,
            "SOUND": sound,
            "HABITAT": habitat,
            "FOOD": food,
            "SYMBOLISM": symbolism,
            "ANIMAL_PERSON": person,
            "ANIMAL_WORKER": worker,
            "ANIMAL_RACE": race,
            "ANIMAL_TEAM": team,
            "ANIMAL_COCKTAIL": cocktail,
            "ANIMAL_CAFE": cafe,
            "ANIMAL_BRAND": brand,
            "ANIMAL_HOTEL": hotel
        }


def get_tags(sentence, combs):
    """
    Generate BIO tags for each word in a sentence based
    on provided combinations.

    Args:
        sentence (str):
            sentence to process
        combs (dict):
            dictionary containing placeholder tags and
            their corresponding values

    Returns:
        list:
            list of tags for the words in the sentence.
    """
    tags = []
    for word in sentence.split():
        if 'ANIMAL' in word:
            tag_key = re.search(r'{.*}', word).group()
            tag_key = tag_key[1:-1]
            tag_values = combs[tag_key].upper().split()

            if len(tag_values) == 1:
                tags.append('B-'+tag_values[0])
            else:
                placeholders = ['I-NOT_ANIMAL'] * len(tag_values)
                placeholders[0] = 'B-NOT_ANIMAL'
                tags.extend(placeholders)
        else:
            tags.append('O')

    return tags


def generate_sentences(sentences, placeholders):
    """
    Generate sentences by replacing placeholders with
    combinations of values

    Args:
        sentences (list):
            list of sentences to process
        placeholders (dict):
            dictionary containing placeholder tags and
            their corresponding values

    Yields:
        tuple:
            tuple containing the modified sentence and its tags
    """
    for sentence in sentences:
        combs = []

        for word in sentence.split():
            for placeholder in placeholders:
                if placeholder in word:
                    pairs = [(placeholder, value) for value in placeholders[placeholder]]
                    combs.append(pairs)

        for combination in product(*combs):
            combination = dict(combination)

            tags = get_tags(sentence, combination)
            modified_sentence = sentence.format(**combination)

            yield (modified_sentence, tags)


if __name__ == '__main__':
    sents_and_tags = list(generate_sentences(sentences, placeholders))

    df = pd.DataFrame(sents_and_tags,
                      columns=['sentence', 'tags']
                      )

    df.to_csv('ner_data.csv',  index=False)
