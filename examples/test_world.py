test_scene = {'1': {'1': '2', '2': '1', '3': '4', '10': '3'},
              '2': {'1': '2', '2': '1', '3': '4', '9': '3'},
              '3': {'1': '2', '2': '1', '3': '4', '4': '3'},
              '4': {'1': '2', '2': '1', '3': '4', '13': '3'},
              '5': {'1': '2', '2': '1', '3': '4', '14': '3'},
              '6': {'1': '2', '2': '1', '3': '4', '6': '3'},
              '7': {'1': '2', '2': '1', '3': '4', '11': '3'}}


test_phrases = [
    "een vrouw met lang haar",
    "een kale man",
    "een man met een bril",
    "een vrouw met oorbellen",
    "de vrouw met oorbellen",
    "een vrouw met een bril",
    "de man met de bril",
    "de kale man",
    "de jongen met de bril",
    "een man met bruin haar en een bril",
    "de vrouw met de oorbellen",
    "de kale man",
    "de vrouw met de oorbellen",
    "de kale man",
    "een oude kale man",
    "de man met de bril",
    "een man met een kuif",
    "de man met de bril",
    "de vrouw met oorbellen",
    "de kale",
    "de vrouw met oorbellen",
    "de kale",
    "een vrouw met bruin haar",
    "de man met de bril",
    "de vrouw met oorbellen",
    "de man met de bril",
    "een vrouw met een bril",
    "de kale"
]

test_phrases_ambiguity = ['man']*28

correct = [10, 3, 2, 1, 1, 9, 2, 3, 2, 4, 1, 3, 1, 3, 13, 2, 14, 2, 1, 3, 1, 3, 6, 2, 1, 2, 11, 3]

introductions = ["daar staat", "dat is", "ik heb", "ik heb daar", "daar heb ik", "op die plek staat"]

test_phrases_5 = [
    "een vrouw met lang haar",
    "een kale man",
    "een jongen met een bril",
    "een vrouw met oorbellen",
    "een man met een kuif"
    "de vrouw met oorbellen",
    "een man met een bril en bruin haar",
    "de jongen met de bril",
    "de kale man",
    "een vrouw met kort bruin haar",
    "de jongen met de bril",
    "een oude vrouw met een bril",
    "de vrouw met de oorbellen",
    "de kale man",
    "een klein kind",
]