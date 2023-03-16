# analyze mention
# if simple deictic, simple pronoun:
#   eliminate gender-mismatch
#   query for recency
# if other mention type:
#   detect definite or indefinite
#   find features in mention (gender, description)
#   find (part of) label used in mention
#   query for labels or features to find matches

from cltl.brain import LongTermMemory
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

m_0 = ['man', 'de man', 'een man', 'met een paardenstaart', 'met de paardenstaart,',
       'met een baard', 'met de baard', 'met een rode paardenstaart',
       'met stijl haar', 'met rood haar', 'met stijl rood haar', 'met het rode haar',
       'met het stijle haar', 'met de rode paardenstaart'
       'de roodharige man', 'een roodharige man', 'met de hoed', 'met een hoed',
       'met een blauwe zwembroek', 'met de blauwe zwembroek']

m_1 = ['man', 'de man', 'een man', 'met kort haar', 'met het korte haar',
       'met stijl haar', 'met het stijle haar', 'met bruin haar',
       'met het bruine haar', 'met kort stijl haar', 'met bruin stijl haar',
       'met het korte stijle haar', 'met het bruine stijle haar',
       'met kort bruin stijl haar', 'met een bril', 'met de bril',
       'de bruinharige man', 'een bruinharige man',
       'met een blauwe zwembroek', 'met de blauwe zwembroek']

m_2 = ['man', 'de man', 'een man', 'de kale man', 'met het kale hoofd',
       'met een kaal hoofd', 'met een baard', 'met de baard', 'met een pet',
       'met de pet', 'met een rode zwembroek', 'met de rode zwembroek']

m_3 = ['vrouw', 'de vrouw', 'een vrouw', 'met een paardenstaart',
       'met de paardenstaart', 'met de grijze paardenstaart',
       'met een grijze paardenstaart', 'met stijl haar', 'met grijs haar',
       'met stijl grijs haar', 'met het grijze haar', 'met het stijle haar',
       'met het stijle grijze haar', 'de grijsharige vrouw',
       'een grijsharige vrouw',
       'met oorbellen', 'met de oorbellen', 'met het bruine badpak',
       'met een bruin badpak']

def model_ambiguity(mention, brain):
    pass


if __name__ == "__main__":
    mention = ["haar"]
    mention2 = ["baardmans"]
    mention3 = ["de man met de baard"]
    descriptions = ["een man", "de man", "met lang blond haar", "met bruin haar", "de vrouw"]
    descriptions2 = ["een andere man met blond haar", "de man met bruin haar"]
    descriptions3 = ["een man met een snor", "de vrouw met de baard", "man met baard", "een jongen met stoppels"]

    mention_embedding = model.encode(mention, convert_to_tensor=True)
    desc_embeddings = model.encode(descriptions, convert_to_tensor=True)

    cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)

    for i in range(len(descriptions)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(mention[0], descriptions[i], cosine_scores[0][i]))

    print()

    mention_embedding = model.encode(mention2, convert_to_tensor=True)
    desc_embeddings = model.encode(descriptions3, convert_to_tensor=True)

    cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)

    for i in range(len(descriptions3)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(mention2[0], descriptions3[i], cosine_scores[0][i]))


