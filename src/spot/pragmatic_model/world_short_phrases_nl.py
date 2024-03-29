ak1 = {'gender': 'vrouw', 'age': 'jong', 'hair_style': 'bob', 'hair_color': 'blond haar', 'hair_type': 'stijl',
       'accessory': 'oorbellen'}
ak2 = {'gender': 'man', 'age': 'jong', 'hair_style': 'kort haar', 'hair_color': 'blond haar', 'hair_type': 'krullen',
       'accessory': 'bril'}
ak3 = {'gender': 'man', 'hair_style': 'kaal', 'hair_color': 'grijs haar',
       'facial_hair': 'baard'}
ak4 = {'gender': 'man', 'hair_style': 'kort haar', 'hair_color': 'bruin haar',
       'hair_type': 'krullen', 'accessory': 'bril'}
ak5 = {'gender': 'man', 'hair_style': 'kaal', 'hair_color': 'bruin haar',
        'facial_hair': 'baard'}
ak6 = {'gender': 'vrouw', 'age': 'jong', 'hair_style': 'bob', 'hair_color': 'zwart haar',
       'hair_type': 'stijl'}
ak7 = {'gender': 'vrouw', 'hair_style': 'lang haar',
        'hair_color': 'blond haar', 'hair_type': 'stijl'}
ak8 = {'gender': 'jongen', 'age': 'kind', 'hair_style': 'kort haar',
        'hair_color': 'blond haar', 'hair_type': 'stijl'}
ak9 = {'gender': 'vrouw', 'age': 'oud', 'hair_style': 'kort haar',
        'hair_color': 'blond haar', 'hair_type': 'stijl', 'accessory': 'bril'}
ak10 = {'gender': 'vrouw', 'age': 'jong', 'hair_style': 'lang haar',
        'hair_color': 'bruin haar', 'hair_type': 'krullen'}
ak11 = {'gender': 'vrouw', 'age': 'jong', 'hair_style': 'lang haar',
        'hair_color': 'bruin haar', 'hair_type': 'stijl', 'accessory': 'bril'}
ak12 = {'gender': 'vrouw', 'hair_style': 'bob',
        'hair_color': 'bruin haar', 'hair_type': 'krullen'}
ak13 = {'gender': 'man', 'age': 'oud', 'hair_style': 'kaal'}
ak14 = {'gender': 'man', 'age': 'jong', 'hair_style': 'kuif',
        'hair_color': 'bruin haar', 'hair_type': 'stijl', 'facial_hair': 'baard'}
ak15 = {'gender': 'man', 'hair_style': 'kort haar',
        'hair_color': 'zwart haar', 'hair_type': 'stekels'}

ak_characters = {'1': ak1, '2': ak2, '3': ak3, '4': ak4, '5': ak5, '6': ak6, '7': ak7,
                 '8': ak8, '9': ak9, '10': ak10, '11': ak11, '12': ak12, '13': ak13, '14': ak14, '15': ak15}
# ak_characters_rounds = {'1': [ak1, ak2, ak3, ak10, ak14], '2': [ak1, ak2, ak3, ak12, ak4], '3': [ak1, ak2, ak3, ak8, ak9],
#                  '4': [ak1, ak2, ak3, ak6, ak15], '5': [ak1, ak2, ak3, ak13, ak11], '6': [ak1, ak2, ak3, ak7, ak5]}
ak_robot_scene = {'1': {'1': '2', '2': '1', '3': '3', '10': '5', '14': '4'},
                  '2': {'1': '4', '2': '2', '3': '1', '12': '5', '4': '4'},
                  '3': {'1': '3', '2': '2', '3': '4', '8': '1', '9': '5'},
                  '4': {'1': '2', '2': '1', '3': '5', '6': '3', '15': '4'},
                  '5': {'1': '2', '2': '4', '3': '3', '13': '5', '11': '1'},
                  '6': {'1': '1', '2': '3', '3': '4', '7': '2', '5': '5'}}

en1 = {'gender': 'man', 'hair_style': 'lang haar',
       'hair_color': 'bruin haar', 'hair_type': 'stijl', 'facial_hair': 'baard'}
en2 = {'gender': 'man', 'hair_style': 'kaal', 'accessory': 'bril'}
en3 = {'gender': 'vrouw', 'hair_style': 'lang haar',
        'hair_color': 'zwart haar', 'hair_type': 'krullen'}
en4 = {'gender': 'man', 'hair_style': 'kort haar',
        'hair_color': 'bruin haar', 'hair_type': 'krullen', 'facial_hair': 'baard'}
en5 = {'gender': 'man', 'hair_style': 'kort haar',
        'hair_color': 'bruin haar', 'hair_type': 'stijl', 'facial_hair': 'snor',
       'accessory': 'bril'}
en6 = {'gender': 'man', 'hair_style': 'kuif',
        'hair_color': 'rood haar', 'accessory': 'bril'}
en7 = {'gender': 'man', 'hair_style': 'knotje',
        'hair_color': 'bruin haar', 'facial_hair': 'baard'}
en8 = {'gender': 'jongen', 'age': 'kind', 'hair_style': 'kort haar',
        'hair_color': 'bruin haar', 'hair_type': 'stijl'}
en9 = {'gender': 'man', 'age': 'oud', 'hair_style': 'kaal',
        'hair_color': 'grijs haar'}
en10 = {'gender': 'vrouw', 'hair_style': 'lang haar',
        'hair_color': 'blond haar', 'hair_type': 'stijl'}
en11 = {'gender': 'vrouw', 'hair_style': 'bob',
        'hair_color': 'blond haar', 'hair_type': 'stijl'}
en12 = {'gender': 'vrouw', 'hair_style': 'lang haar',
        'hair_color': 'bruin haar', 'hair_type': 'stijl'}

en_characters = [en1, en2, en3, en4, en5, en6, en7, en8, en9, en10, en11, en12]
en_characters_rounds = {'1': [en1, en2, en3, en11, en12], '2': [en1, en2, en3, en4, en6], '3': [en1, en2, en3, en8, en9],
                 '4': [en1, en2, en3, en6, en12], '5': [en1, en2, en3, en10, en7], '6': [en1, en2, en3, en5, en12]}

