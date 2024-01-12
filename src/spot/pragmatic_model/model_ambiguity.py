import math
from enum import Enum
from scipy.stats import entropy
from random import choice

import torch
from sentence_transformers import SentenceTransformer, util
from world_short_phrases_nl import ak_characters, ak_robot_scene
from gensim.models import word2vec, KeyedVectors
import numpy as np
from collections import Counter

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
filename = '/Users/jaapkruijt/Documents/GitHub/spot_disambiguation_model/local_files/nlwiki_20180420_300d.txt'
# wiki2vec = KeyedVectors.load_word2vec_format(filename)
entity_mention_history = {}
entity_history = []
recencies = {str(i+1): 0 for i in range(len(ak_characters))}


class SimilarityScorer:
    def calculate_sentence_simscores(self, source: list, descriptions: list):
        """
        Compare one list of strings against another string using cosine similarity measure
        :return: similarity score for each pair
        """
        source_embeddings = model.encode(source)
        description_embedding = model.encode(descriptions)
        cosine_scores = util.cos_sim(source_embeddings, description_embedding)

        return cosine_scores

    # def calculate_word_simscores(self, source: list, descriptions: list):
    #     for attribute in descriptions:
    #         for word in source:
    #             similarity = wiki2vec.similarity(attribute, word)

    def calculate_textual_overlap(self, source, descriptions):
        textual_match = []

        for phrase in descriptions:
            if phrase in source:
                textual_match.append(phrase)

        return textual_match


class DisambiguatorStatus(Enum):
    AWAIT_NEXT = 1
    SUCCESS = 2
    NO_MATCH = 3
    MATCH_MULTIPLE = 4
    MATCH_PREVIOUS = 5


class Disambiguator:
    def __init__(self, world, scenes):
        self.status = DisambiguatorStatus.AWAIT_NEXT
        self.common_ground = CommonGround()
        self.lexicon = Lexicon()
        self.scorer = SimilarityScorer()
        self.world = world
        self.scenes = scenes
        self.scene_characters = []

    def start_game(self):
        self.lexicon.compute_base_lexicon(self.world)
        self.scene_characters = list(self.scenes['1'].keys())
        self.lexicon.rsa(self.lexicon.base_lexicon(), self.scene_characters)

    def advance_round(self, round_number):
        self.scene_characters = list(self.scenes[str(round_number)].keys())
        self.lexicon.rsa(self.lexicon.base_lexicon(), self.scene_characters)
        self.status = DisambiguatorStatus.AWAIT_NEXT

        return self.lexicon.pragmatic_lexicon()

    def advance_position(self):
        self.status = DisambiguatorStatus.AWAIT_NEXT

    def disambiguate(self, mention, approach='full', threshold=0.6):
        literal_candidate_attributes, literal_candidate_scores = self.literal_match(mention)
        scores = np.array(list(literal_candidate_scores.values()))
        max_score = scores.max(initial=0)
        score_entropy = entropy(scores, base=10)

        if max_score == 0:
            self.status = DisambiguatorStatus.NO_MATCH
            return "No match found"

        top_candidates = []
        for candidate, score in literal_candidate_scores.items():
            if score == max_score:
                top_candidates.append(candidate)

        if len(top_candidates) == 1:
            self.status = DisambiguatorStatus.SUCCESS
            return top_candidates[0]

        if len(top_candidates) > 1:
            self.status = DisambiguatorStatus.MATCH_MULTIPLE
            differences = self.find_differences(top_candidates)
            if approach == 'full':
                pragmatic_match = self.contextual_pragmatic_match(literal_candidate_attributes)
            return differences

    def find_differences(self, candidates):
        unique_attributes = {candidate: [] for candidate in candidates}
        for attribute, characters in self.lexicon.base_lexicon().items():
            options = [character for character in characters if characters[character] == 1]
            overlap = [character for character in candidates if character in options]
            if len(overlap) == 1:
                unique_attributes[overlap[0]].append(attribute)

        return unique_attributes

    def literal_match(self, mention, threshold=0.6, approach='nli'):
        # TODO make this compatible with all literal scoring approaches
        """
        compares the mention with descriptions for each character
        :param mention: string
        :param threshold: float
        :param approach: string
        :return: scores for each mention_character pair
        """
        candidate_attributes = {character: set() for character in self.scene_characters}
        candidate_scores = {character: 0 for character in self.scene_characters}
        if len(mention.split()) >= 3:
            mention_parts = split_on_window(mention)
        else:
            mention_parts = [mention]
        scoring = self.scorer.calculate_sentence_simscores(mention_parts, list(self.lexicon.base_lexicon().keys()))
        scoring = scoring.tolist()
        matches = []
        for section in scoring:
            matches.append(list(zip(section, list(self.lexicon.base_lexicon().keys()))))
        for match in matches:
            for score, attribute in match:
                if score >= threshold:
                    for character, value in self.lexicon.base_lexicon()[attribute].items():
                        if value == 1 and character in self.scene_characters:
                            candidate_attributes[character].add(attribute)
                            candidate_scores[character] += score

        score_sum = sum(list(candidate_scores.values()))
        if score_sum > 0:
            for candidate, score in candidate_scores.items():
                candidate_scores[candidate] = score/score_sum

        return candidate_attributes, candidate_scores

    def contextual_pragmatic_match(self, candidates):
        total = {candidate: 0 for candidate in candidates.keys()}
        for candidate, attributes in candidates.items():
            for attribute in attributes:
                probability = self.lexicon.pragmatic_lexicon()[attribute][candidate]
                total[candidate] += probability

        return total

    def mention_history_scoring(self, mention):
        mention_embedding = model.encode([mention], convert_to_tensor=True)
        previous_mention_score = {character: 0 for character in self.common_ground.history}
        for character, previous_mentions in self.common_ground.history.items():
            if previous_mentions:
                prev_embeddings = model.encode(previous_mentions, convert_to_tensor=True)
                cosine_scores = util.cos_sim(mention_embedding, prev_embeddings)
                cosine_mean = torch.mean(cosine_scores)
                previous_mention_score[character] = cosine_mean

        return previous_mention_score


class CommonGround:
    def __init__(self):
        self.history = {}
        self.scene = {}
        self.priors = {}


class Lexicon:
    def __init__(self):
        self._base_lexicon = {}
        self._pragmatic_lexicon = {}

    def base_lexicon(self):
        return self._base_lexicon

    def pragmatic_lexicon(self):
        return self._pragmatic_lexicon

    def get_attributes(self, characters):
        """
        Loop over characters,
        Get attribute description from each category,
        :return: dict with one description per character
        """
        attributes = {}

        for character in characters:
            for attribute in character.values():
                attributes[attribute] = {}

        return attributes

    def compute_base_lexicon(self, characters):
        """
        :param characters: list
        :return:
        """
        lexicon = self.get_attributes(characters)

        for i, character in enumerate(characters):
            for attribute in lexicon:
                if attribute in character.values():
                    lexicon[attribute][str(i + 1)] = 1
                else:
                    lexicon[attribute][str(i + 1)] = 0

        self._base_lexicon = lexicon

    def literal_listener(self, lexicon, characters):
        uniform_prior = 1 / len(characters)
        listener_probs = {}
        for word, referents in lexicon.items():
            listener_probs[word] = {}
            for character in characters:
                value = referents[character]
                if value == 1:
                    listener_probs[word][character] = uniform_prior
            probability_sum = sum(list(listener_probs[word].values()))
            for referent, probability in listener_probs[word].items():
                listener_probs[word][referent] = probability / probability_sum
            # print(sum(list(literal_probs[word].values())))

        return listener_probs

    def pragmatic_speaker(self, literal_probability, characters, log_base=10, alpha=1):
        speaker_probs = {}
        words = []
        for character in characters:
            speaker_probs[character] = {}
            for word, referents in literal_probability.items():
                words.append(word)
                if character in referents.keys():
                    word_probability = math.exp(math.log(alpha * referents[character], log_base))
                    speaker_probs[character][word] = word_probability
            probability_sum = sum(list(speaker_probs[character].values()))
            for word, probability in speaker_probs[character].items():
                speaker_probs[character][word] = probability / probability_sum
            # print(sum(list(speaker_probs[character].values())))

        return speaker_probs

    def pragmatic_listener(self, speaker_probability, characters):
        listener_probs = {}
        uniform_prior = 1 / len(characters)
        for character, attributes in speaker_probability.items():
            for attribute in attributes:
                if attribute not in listener_probs.keys():
                    listener_probs[attribute] = {}
                listener_probs[attribute][character] = attributes[attribute] * uniform_prior
        for word, referents in listener_probs.items():
            probability_sum = sum(list(referents.values()))
            for referent, probability in referents.items():
                listener_probs[word][referent] = probability / probability_sum
            # print(sum(list(listener_probs[word].values())))

        return listener_probs

    def rsa(self, lexicon, round_characters, log_base=10, alpha=1):
        l_lit = self.literal_listener(lexicon, round_characters)
        s_prag = self.pragmatic_speaker(l_lit, round_characters, log_base, alpha)
        l_prag = self.pragmatic_listener(s_prag, round_characters)

        self._pragmatic_lexicon = l_prag


# taken from stackoverflow
def split_on_window(sequence, limit=3):
    """

    :param sequence:
    :param limit:
    :return:
    """
    results = []
    split_sequence = sequence.split()
    iteration_length = len(split_sequence) - (limit - 1)
    max_window_indices = range(iteration_length)
    for index in max_window_indices:
        results.append(' '.join(split_sequence[index:index + limit]))
    return results


def normalize(value_dict):
    value_sum = sum(list(value_dict.values()))
    if value_sum > 0:
        for key, value in value_dict.items():
            value_dict

def lexicon_update():
    """
    update character lexicon to account for new descriptions used and update low-scoring lexical items
    :return:
    """
    pass


def history_update():
    """
    update history of character mentions
    :return:
    """
    pass


def character_priors():
    pass


def common_ground_update():
    """
    history_update + lexicon_update + character priors
    :return:
    """
    previous_rounds = {}
    current_round = {}
    mention_history = {}
    lexicon = {}
    priors = {}


def rank_by_recency(entity_recencies, characters):
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





if __name__ == "__main__":
    ment = 'een oudere dame met een bril op.'
    ment2 = 'een vrouw met blond haar'
    disambiguator = Disambiguator(ak_characters, ak_robot_scene)
    disambiguator.start_game()
    rational = disambiguator.advance_round('3')
    print(rational)
    result = disambiguator.disambiguate(ment2)
    print(result)
    # lex = compute_base_lexicon(ak_characters)
    # # chars = ['1', '2', '3', '13', '11']
    # # literal_listener = literal_listener(lex, chars)
    # # speaker = pragmatic_speaker(literal_listener, chars)
    # # pragmatic_listener = pragmatic_listener(speaker, chars)
    # # print(pragmatic_listener)
    # candidate_list = literal_match(ment, lex)
    # print(candidate_list)
    # item = 'met bril'
    # print(calculate_simscores([ment], [item]))




