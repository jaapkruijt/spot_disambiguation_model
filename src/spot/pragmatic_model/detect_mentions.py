# take utterance string
# truncate introduction
# detect positional description/adverb
# detect conjunctions
# isolate mention(s)

# import spacy
import re
from thefuzz import fuzz

# nlp = spacy.load('nl_core_news_sm')

positions = ['op', 'plek', 'twee', 'drie', 'vier', 'vijf', 'eerste', 'tweede', 'derde', 'vierde',
             'vijfde', 'dan', 'daarnaast', '1', '2', '3', '4', '5', 'staat', 'plaatje', 'foto', 'nummer', 'de volgende',
             'is', 'dat is', 'die is', 'ik', 'heb', 'bij', 'mij', 'was', 'het volgende', 'daar', 'laatste', 'plekje',
             'in het midden', 'links', 'rechts', 'helemaal', 'als eerste', 'als tweede', 'als derde', 'als vierde',
             'als vijfde', 'als laatste', 'we', 'hebben', 'aan de linkerkant', 'aan de rechterkant',
             'van links', 'van rechts', 'er', 'op de']

# Spacy: laatste NP in de utterance?
# split utterance per zin
# agreement = ["ook", "niet", "zelfde", "andere", "anders"]
# voor reactie robot: 'oh ja', 'even kijken' etc
# na vraag robot: 'wie staat er bij jou op...', 'en dan?', 'en de volgende' 'en de laatste', 'wie staat waar?', etc
# soms volgt een uitweiding van de beschrijving - onderdeel van de mention
# state tracking: zijn we in een ronde van het spel?
# return introductie en mention

robot_responses = ['Eens kijken', 'Even kijken hoor, eh', 'Even kijken hoe dat er bij mij uitziet.',
                   'Eh, even denken', 'Oh ja, die staat bij mij op plek', 'Die staat bij mij ook op die plek.',
                   'Bij mij staat die op plek', 'Die staat bij mij ergens anders, namelijk de plek',
                   'Oh, dan staat die op een andere plek', 'Oh ja, die staat bij ons allebei op dezelfde plek']
robot_questions = ['En de volgende?', 'wie staat er bij jou op de eerste plek?', 'wie staat waar?',
                   'wat zie je nu?', 'en dan?', 'en de volgende?', 'en de laatste?', 'en daarna?',]


def detect_mentions(utterances):
    for i, utterance in enumerate(utterances):
        if utterances[i-1]['text'] in robot_questions:
            intro = []
            for word in positions:
                if positions in utterance['text']:
                    re.sub(word, '', utterance['text'])
                    intro.append(word)
            utterance['mention'] = utterance['text']
        elif utterances[i+1]['text'] in robot_responses:
            intro = []
            for word in positions:
                if positions in utterance['text']:
                    re.sub(word, '', utterance['text'])
                    intro.append(word)
            utterance['mention'] = utterance['text']
        elif utterances[i-1]['contains_mention']:
            intro = []
            for word in positions:
                if positions in utterance['text']:
                    re.sub(word, '', utterance['text'])
                    intro.append(word)
            utterance['mention'] = utterance['text']




if __name__ == "__main__":
    detect_mentions()
