# analyze mention
# if simple deictic, simple pronoun:
#   eliminate gender-mismatch
#   query for recency
# if other mention type:
#   detect definite or indefinite
#   find features in mention (gender, description)
#   find (part of) label used in mention
#   query for labels or features to find matches

import torch
from cltl.brain import LongTermMemory
from sentence_transformers import SentenceTransformer, util
from world import characters

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def calculate_average_simscores(mention, character_descriptions: dict):
    mention_embedding = model.encode([mention], convert_to_tensor=True)
    character_scores = character_descriptions.copy()
    for character in character_scores:
        character_scores[character] = 0
    for character, features in character_descriptions.items():
        descriptors = []
        for feature, descriptions in features.items():
            descriptors.extend(descriptions)
        desc_embeddings = model.encode(descriptors, convert_to_tensor=True)
        cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)
        cosine_mean = torch.mean(cosine_scores)
        # avg_score = float(cosine_sum)/len(descriptors)
        character_scores[character] = float(cosine_mean)

    return character_scores


def calculate_simscores_per_feature(mention, character_descriptions: dict):
    mention_embedding = model.encode([mention], convert_to_tensor=True)
    character_scores = character_descriptions.copy()
    for character in character_scores:
        character_scores[character] = 0
    for character, features in character_descriptions.items():
        scores = []
        for feature, description in features.items():
            if description:
                desc_embeddings = model.encode(description, convert_to_tensor=True)
                cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)
                cosine_mean = torch.mean(cosine_scores)
                scores.append(cosine_mean)
        concat_scores = torch.stack(scores)
        mean_scores = torch.mean(concat_scores)
        character_scores[character] = mean_scores

    return character_scores


if __name__ == "__main__":
    current_mention = 'die met een paardenstaart'
    alt_mention1 = 'niet die met een paardenstaart'

    scores = calculate_simscores_per_feature(current_mention, characters)
    mean = calculate_average_simscores(current_mention, characters)
    print(scores)
    print(mean)

    # mention = ["haar"]
    # mention2 = ["baardmans"]
    # mention3 = ["de man met de baard"]
    # descriptions = ["een man", "de man", "met lang blond haar", "met bruin haar", "de vrouw"]
    # descriptions2 = ["een andere man met blond haar", "de man met bruin haar"]
    # descriptions3 = ["een man met een snor", "de vrouw met de baard", "man met baard", "een jongen met stoppels"]
    #
    # mention_embedding = model.encode(mention, convert_to_tensor=True)
    # desc_embeddings = model.encode(descriptions, convert_to_tensor=True)
    #
    # cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)
    #
    # for i in range(len(descriptions)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(mention[0], descriptions[i], cosine_scores[0][i]))
    #
    # print()
    #
    # mention_embedding = model.encode(mention2, convert_to_tensor=True)
    # desc_embeddings = model.encode(descriptions3, convert_to_tensor=True)
    #
    # cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)
    # cosine_sum = torch.sum(cosine_scores)
    # print(float(cosine_sum))
    # print(cosine_sum)
    #
    # for i in range(len(descriptions3)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(mention2[0], descriptions3[i], cosine_scores[0][i]))


