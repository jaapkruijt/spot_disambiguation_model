test_scene = {'1': {'1': '2', '2': '1', '3': '4', '14': '3'},
              '2': {'1': '2', '2': '1', '3': '4', '9': '3'},
              '3': {'1': '2', '2': '1', '3': '4', '4': '3'},
              '4': {'1': '2', '2': '1', '3': '4', '13': '3'}}


test_phrases = [
    "een man met een kuif",
    "een kale man",
    "een jongen met een bril",
    "een vrouw met oorbellen",
    "de vrouw met oorbellen",
    "een vrouw met een bril",
    "de jongen met de bril",
    "de kale man",
    "de jongen met de bril",
    "een man met bruin haar en een bril",
    "de vrouw met de oorbellen",
    "de kale man",
    "de vrouw met de oorbellen",
    "de kale man",
    "een oude kale man",
    "de jongen met de bril"
]

correct = [14, 3, 2, 1, 1, 9, 2, 3, 2, 4, 1, 3, 1, 3, 13, 2]

introductions = ["daar staat", "dat is", "ik heb", "ik heb daar", "daar heb ik"]