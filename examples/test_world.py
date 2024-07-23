test_scene = {'1': {'1': '2', '2': '1', '3': '4', '10': '3'},
              '2': {'1': '2', '2': '1', '3': '4', '9': '3'},
              '3': {'1': '2', '2': '1', '3': '4', '4': '3'},
              '4': {'1': '2', '2': '1', '3': '4', '13': '3'},
              '5': {'1': '2', '2': '1', '3': '4', '14': '3'},
              '6': {'1': '2', '2': '1', '3': '4', '6': '3'},
              '7': {'1': '2', '2': '1', '3': '4', '11': '3'}}


test_phrases = [
    "A woman with long hair",
    "A bald man",
    "A man with glasses",
    "A woman with earrings",
    "The woman with the earrings",
    "A woman with glasses",
    "The man with glasses",
    "The bald man",
    "The old man with glasses",
    "A man with brown hair and glasses",
    "The woman with earrings",
    "The bald man",
    "The woman with earrings",
    "The bald man",
    "An old bald man",
    "The man with glasses",
    "A man with a slick back",
    "The man with glasses",
    "The woman with earrings",
    "The bald one",
    "The woman with earrings",
    "Baldie",
    "A woman with brown hair",
    "The man with glasses",
    "The woman with earrings",
    "The man with glasses",
    "A woman with glasses",
    "Baldie"
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