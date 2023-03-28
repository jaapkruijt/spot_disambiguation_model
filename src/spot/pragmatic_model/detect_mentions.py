# take utterance string
# truncate introduction
# detect positional description/adverb
# detect conjunctions
# isolate mention(s)

import spacy

nlp = spacy.load('nl_core_news_sm')

introductions = ["ik", "jij", "heb", "hebt", "bij", "jou", "mij", "dan"]
positions = ["naast", "links", "rechts", "daarnaast", "daarvan", "van", "staat", "staan", "daar"]
agreement = ["ook", "niet", "zelfde", "andere", "anders"]

def detect_mentions(utterance: str):
    pass



def analyze_mention(mention: dict):
    doc = nlp(mention['label'])
    mention['features'] = []
    for token in doc:
        if 'head_pos' not in mention:
            head_pos = token.head.pos_
            mention['head_pos'] = head_pos
        if token.pos_ == 'PRON':
            gender = token.tag_.split('|')[-1] if 'Prs' in token.morph.get("PronType") else 'Undefined'
            mention['gender'] = gender
        elif token.pos_ == 'DET':
            det = token.morph.get("Definite")[0] if token.morph.get("Definite") else 'Def'
            mention['determiner'] = det
        elif token.pos_ in ['NOUN', 'ADJ', 'VERB']:
            mention['features'].append(token.text)

    return mention


if __name__ == "__main__":
    text = "de man met blond haar"
    ment = {'label': text}
    ment = analyze_mention(ment)
    print(ment)

    # dezelfde, diegene, hetzelfde,
    # constituency parsing
