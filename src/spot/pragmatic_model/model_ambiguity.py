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
entity_mention_history = {}
entity_history = []
recencies = {character: 0 for character in characters}


def calculate_average_simscores(mention, character_descriptions: dict):
    mention_embedding = model.encode([mention], convert_to_tensor=True)
    character_scores = {character: 0 for character in character_descriptions}
    for character, features in character_descriptions.items():
        descriptors = []
        for feature, descriptions in features.items():
            descriptors.extend(descriptions)
        desc_embeddings = model.encode(descriptors, convert_to_tensor=True)
        cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)
        cosine_mean = torch.mean(cosine_scores)
        # avg_score = float(cosine_sum)/len(descriptors)
        character_scores[character] = cosine_mean

    return character_scores


def calculate_simscores_per_feature(mention, character_descriptions: dict):
    mention_embedding = model.encode([mention], convert_to_tensor=True)
    character_scores = {character: 0 for character in character_descriptions}
    for character, features in character_descriptions.items():
        scores = []
        for feature, description in features.items():
            if description:
                desc_embeddings = model.encode(description, convert_to_tensor=True)
                cosine_scores = util.cos_sim(mention_embedding, desc_embeddings)
                cosine_mean = torch.mean(cosine_scores)
                scores.append(cosine_mean)
        concat_scores = torch.stack(scores)
        mean_score = torch.mean(concat_scores)
        character_scores[character] = mean_score

    return character_scores


def rank_by_recency(entity_recencies):
    character_prior = {character: 1 / len(characters) for character in characters}
    for entity, recency in entity_recencies.items():
        recency_score = 1/(recency+1)
        character_prior[entity] += recency_score
    factor = 1.0 / sum(character_prior.values())
    for character in character_prior:
        character_prior[character] = character_prior[character] * factor
    # recency_array = np.array(list(character_prior.values()))
    # normalized_recencies = recency_array / sum(recency_array)
    # character_priors = {character: prior for character in characters for prior in normalized_recencies}

    return character_prior


def previous_mention_scoring(mention_history, mention):
    mention_embedding = model.encode([mention], convert_to_tensor=True)
    previous_mention_score = {character: 0 for character in mention_history}
    for character, previous_mentions in mention_history.items():
        if previous_mentions:
            prev_embeddings = model.encode(previous_mentions, convert_to_tensor=True)
            cosine_scores = util.cos_sim(mention_embedding, prev_embeddings)
            cosine_mean = torch.mean(cosine_scores)
            previous_mention_score[character] = cosine_mean

    return previous_mention_score


if __name__ == "__main__":
    current_mention = 'baardmans'
    alt_mention1 = 'niet die met een paardenstaart'

    recencies = {'m_0': 1, 'm_1': 2, 'm_2': 3, 'm_3': 4}

    priors = rank_by_recency(recencies)
    print(priors)

    scores = calculate_simscores_per_feature(current_mention, characters)
    mean = calculate_average_simscores(current_mention, characters)
    print(scores)
    print(mean)

    # emb1 = model.encode(['liefde'], convert_to_tensor=True)
    # emb2 = model.encode(['bril'], convert_to_tensor=True)
    # cosine_score = util.cos_sim(emb1, emb2)
    # print(cosine_score)


