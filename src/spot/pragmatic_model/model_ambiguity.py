import math
from enum import Enum
import logging
from scipy.stats import entropy
from scipy.special import rel_entr
import spacy
import random
from random import choice

import torch
from sentence_transformers import SentenceTransformer, util
from spot.pragmatic_model.world_short_phrases_nl import ak_characters, ak_robot_scene
from gensim.models import word2vec, KeyedVectors
import numpy as np
import json
from spot.pragmatic_model.detect_mentions import subtree_right_approach
from collections import Counter

# nlp = spacy.load('nl_core_news_lg')
# model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
w2vfile = '/Users/jaapkruijt/Documents/GitHub/spot_disambiguation_model/local_files/nlwiki_20180420_300d.txt'
# wiki2vec = KeyedVectors.load_word2vec_format(filename)
entity_mention_history = {}
entity_history = []
recencies = {str(i+1): 0 for i in range(len(ak_characters))}


class SimilarityScorer:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1'):
        self.model = SentenceTransformer(model_name)

    def calculate_sentence_simscores(self, source: list, descriptions: list):
        """
        Compare one list of strings against another string using cosine similarity measure
        :return: similarity score for each pair
        """
        source_embeddings = self.model.encode(source)
        description_embedding = self.model.encode(descriptions)
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
    SUCCESS_HIGH = 2
    SUCCESS_LOW = 3
    NO_MATCH = 4
    MATCH_MULTIPLE = 5
    MATCH_PREVIOUS = 6
    NEG_RESPONSE = 7


class Disambiguator:
    def __init__(self, world, scenes, high_engagement=bool):
        self._status = DisambiguatorStatus.AWAIT_NEXT
        self.common_ground = CommonGround()
        self.lexicon = Lexicon()
        self.scorer = SimilarityScorer()
        self.world = world
        self.scenes = scenes
        self.scene_characters = []
        self.current_round = 0
        self.high_engagement = high_engagement

    def status(self):
        return self._status.name

    def advance_round(self, round_number=None, start=False):
        if start:
            self.lexicon.compute_base_lexicon(list(self.world.values()))
        else:
            self.confirm_character_position()
        self.current_round += 1
        if not round_number:
            round_number = self.current_round
        self.scene_characters = list(self.scenes[str(round_number)].keys())
        self.lexicon.rsa(self.lexicon.base_lexicon(), self.scene_characters)
        self._status = DisambiguatorStatus.AWAIT_NEXT
        self.common_ground.current_position = 1
        self.common_ground.update_priors(self.scene_characters)
        self.common_ground.positions_discussed[str(round_number)] = {}
        self.common_ground.reset_under_discussion()

        return self.lexicon.pragmatic_lexicon()

    def advance_position(self, position=None):
        self.confirm_character_position()
        self._status = DisambiguatorStatus.AWAIT_NEXT
        if position:
            self.common_ground.current_position = position
        else:
            self.common_ground.current_position += 1
        self.common_ground.reset_under_discussion()

    def confirm_character_position(self):
        selection = self.common_ground.under_discussion['guess'][-1]
        mention = self.common_ground.under_discussion['mention'][-1]
        logging.debug('Selection: %s', selection)
        logging.debug('Mention added to history: %s', mention)
        if selection in self.common_ground.history.keys():
            self.common_ground.history[selection]['human'].append(mention)
        else:
            self.common_ground.history[selection] = {'human': [mention]}
        # TODO do we want to add the robot response to mention history?
        if self.high_engagement:
            response = self.common_ground.under_discussion['response'][-1]
            if response:
                if 'robot' in self.common_ground.history[selection].keys():
                    self.common_ground.history[selection]['robot'].append(response)
                else:
                    self.common_ground.history[selection]['robot'] = [response]
        self.common_ground.positions_discussed[str(self.current_round)][self.common_ground.current_position] = selection
        self.common_ground.update_priors(self.scene_characters, character_mentioned=selection)

    def disambiguate(self, mention, approach='full', use_history=True, test=False, literal_threshold=0.7,
                     history_threshold=0.80, split_size=2, certainty_threshold=0.60):
        """

        :param mention: string
        :param approach: 'full' or 'literal'
        :param use_history: bool
        :param test: bool
        :param literal_threshold: float
        :param history_threshold: float
        :param split_size: int
        :param certainty_threshold: float
        :return: selection, certainty, position, difference
        """
        mention.strip()
        mention.strip('.')
        logging.debug("Mention: %s", mention)
        logging.debug("Status: %s", self.status())

        # check if repair, positive response or new input
        if self.status() == 'MATCH_MULTIPLE':
            if 'ja' in mention.lower():
                self._status = DisambiguatorStatus.SUCCESS_HIGH
                selected = self.common_ground.under_discussion['guess'][-1]
                position = self.common_ground.under_discussion['position'][-1]

                return selected, 1.0, int(position), None

        # compute similarity scores with history and attributes
        history_score = self.mention_history_scoring(mention, history_threshold)
        literal_candidate_attributes, literal_candidate_scores = self.literal_match(mention, threshold=literal_threshold, split_size=split_size)
        for character, attributes in literal_candidate_attributes.items():
            logging.debug("Attributes for character %s: %s", character, attributes)
        for character, score in literal_candidate_scores.items():
            prior = self.common_ground.priors[character]
            literal_candidate_scores[character] *= prior
        if use_history:
            for character, scores in history_score.items():
                for score in scores:
                    if score > history_threshold:
                        # TODO refine approach
                        literal_candidate_scores[character] += 0.1
        # lower probability for original guess in case of negative feedback
        if self.status() in ['MATCH_PREVIOUS', 'MATCH_PREVIOUS', 'SUCCESS_LOW']:
            # TODO check if too thorough
            for previous_guess in self.common_ground.under_discussion['guess']:
                literal_candidate_scores[previous_guess] -= 0.1
                if literal_candidate_scores[previous_guess] < 0.0:
                    literal_candidate_scores[previous_guess] = 0.0
        literal_candidate_scores = normalize(literal_candidate_scores)
        scores = np.array(list(literal_candidate_scores.values()))
        logging.debug("Score distribution: %s", scores)
        max_score = scores.max(initial=0)
        selected = '0'

        # case: no match
        if max_score == 0.0:
            if 'nee' in mention.lower():
                self._status = DisambiguatorStatus.NEG_RESPONSE
                return selected, 1.0, None, None
            self._status = DisambiguatorStatus.NO_MATCH
            self.common_ground.add_under_discussion(mention)
            return selected, 1.0, None, None

        # find top candidate and compute certainty
        # uniform = uniform = np.random.uniform(size=len(scores))
        uniform = [1/len(scores)]*len(scores)
        logging.debug("Uniform distribution: %s", uniform)
        certainty = sum(rel_entr(scores, uniform))/math.log(len(scores))
        logging.debug("Certainty according to KL: %s", certainty)
        # score_entropy = entropy(scores, base=2)
        # logging.debug("Score entropy: %s", score_entropy)
        # TODO check certainty/entropy: try KL divergence?
        # certainty = 1-(score_entropy/math.log(5, 2))
        top_candidates = []
        for candidate, score in literal_candidate_scores.items():
            if score == max_score:
                top_candidates.append(candidate)

        # case: one top candidate
        if len(top_candidates) == 1:
            selected = top_candidates[0]
            if selected in self.common_ground.positions_discussed[str(self.current_round)].values():
                self._status = DisambiguatorStatus.MATCH_PREVIOUS

                position = self.scenes[str(self.current_round)][selected]
                self.common_ground.add_under_discussion(mention, selected, position)
                return selected, certainty, int(position), None
            else:
                if certainty > certainty_threshold:
                    self._status = DisambiguatorStatus.SUCCESS_HIGH
                    position = self.scenes[str(self.current_round)][selected]
                    # TODO breaks here if only match from history
                    response = self.format_response_phrase(self.world[selected]['gender'],
                                                           random.choice(list(literal_candidate_attributes[selected])))
                    self.common_ground.add_under_discussion(mention, selected, position, response)
                    return selected, certainty, int(position), response
                else:
                    self._status = DisambiguatorStatus.SUCCESS_LOW
                    response, selected = self.find_and_select_differences(self.scene_characters,
                                                                          single_candidate=selected)
                    position = self.scenes[str(self.current_round)][selected]
                    self.common_ground.add_under_discussion(mention, selected, position, response)
                    return selected, certainty, int(position), response

        # case: more than one top candidate
        if len(top_candidates) > 1:
            self._status = DisambiguatorStatus.MATCH_MULTIPLE
            if approach == 'full':
                top_candidate_attributes = {candidate: attributes for candidate, attributes
                                            in literal_candidate_attributes.items() if candidate in top_candidates}
                pragmatic_match = self.contextual_pragmatic_match(top_candidate_attributes)
                ordered = dict(sorted(pragmatic_match.items(), key=lambda item: item[1], reverse=True))
                logging.debug("Ordered dict: %s", ordered)
                selected = next(iter(ordered))
                logging.debug('Selected character: %s', selected)
                response, selected = self.find_and_select_differences(top_candidates, single_candidate=selected)
                scores = np.array(list(ordered.values()))
                uniform = [1/len(scores)]*len(scores)
                logging.debug("Uniform distribution: %s", uniform)
                certainty = sum(rel_entr(scores, uniform))/math.log(len(scores))
                # score_entropy = entropy(scores, base=2)
                # certainty = 1-(score_entropy/math.log(len(ordered), 2))
                position = self.scenes[str(self.current_round)][selected]
            else:
                response, selected = self.find_and_select_differences(top_candidates)
                position = self.scenes[str(self.current_round)][selected]

            self.common_ground.add_under_discussion(mention, selected, position, response)

            return selected, certainty, int(position), response

    def find_and_select_differences(self, candidates, single_candidate=None):
        unique_attributes = {candidate: [] for candidate in candidates}
        for attribute, characters in self.lexicon.base_lexicon().items():
            options = [character for character in characters if characters[character] == 1]
            overlap = [character for character in candidates if character in options]
            if len(overlap) == 1:
                unique_attributes[overlap[0]].append(attribute)

        if single_candidate:
            candidate_guess = single_candidate
            if unique_attributes[single_candidate]:
                logging.debug("Unique features: %s", unique_attributes[single_candidate])
                difference = random.choice(unique_attributes[single_candidate])
            else:
                attribute_options = [attribute for feature, attribute in self.world[candidate_guess].items()
                                     if feature != 'gender']
                difference = random.choice(attribute_options)
                logging.debug("No unique features found")
        else:
            candidate_guess = random.choice(candidates)
            logging.debug("Random guess: %s", candidate_guess)
            if unique_attributes[candidate_guess]:
                logging.debug("Unique features for guess: %s", unique_attributes[candidate_guess])
                difference = random.choice(unique_attributes[candidate_guess])
            else:
                attribute_options = [attribute for feature, attribute in self.world[candidate_guess].items()
                                     if feature != 'gender']
                difference = random.choice(attribute_options)
                logging.debug("No differences found")

        phrase = self.format_response_phrase(self.world[candidate_guess]['gender'], difference)

        return phrase, candidate_guess

    def literal_match(self, mention, threshold=0.6, approach='nli', split_size=2):
        # TODO make this compatible with all literal scoring approaches
        """
        compares the mention with descriptions for each character
        :param mention: string
        :param threshold: float
        :param approach: string
        :return: scores for each mention_character pair
        """
        # create dictionaries
        candidate_attributes = {character: set() for character in self.scene_characters}
        candidate_scores = {character: 0 for character in self.scene_characters}
        # create mention chunks of size 'split_size' using sliding window
        if len(mention.split()) >= split_size:
            mention_parts = split_on_window(mention, limit=split_size)
        else:
            mention_parts = [mention]
        # log information about mention parts
        for part in mention_parts:
            logging.debug("Mention part: %s", part)
        # score mention parts against attributes
        scoring = self.scorer.calculate_sentence_simscores(mention_parts, list(self.lexicon.base_lexicon().keys()))
        scoring = scoring.tolist()
        matches = []
        for section in scoring:
            matches.append(list(zip(section, list(self.lexicon.base_lexicon().keys()))))
        mention_text = {}
        for i, match in enumerate(matches):
            for score, attribute in match:
                if score >= threshold:
                    logging.debug("Match for mention part %s with %s, score: %s", mention_parts[i], attribute, score)
                    mention_text[mention_parts[i]] = None
                    for character, value in self.lexicon.base_lexicon()[attribute].items():
                        if value == 1 and character in self.scene_characters:
                            # avoid multiple mentions
                            if attribute not in candidate_attributes[character]:
                                # logging.debug("Found match for attribute %s for character %s", attribute, character)
                                candidate_attributes[character].add(attribute)
                                candidate_scores[character] += score

        candidate_scores = normalize(candidate_scores)

        mention_string = ''
        for i, part in enumerate(mention_text):
            if i == 0:
                mention_string += part
            else:
                mention_string += ' ' + part.split()[-1]

        return candidate_attributes, candidate_scores

    def contextual_pragmatic_match(self, candidates):
        total = {candidate: 0 for candidate in candidates.keys()}
        for candidate, attributes in candidates.items():
            for attribute in attributes:
                probability = self.lexicon.pragmatic_lexicon()[attribute][candidate]
                total[candidate] += probability

        return total

    def mention_history_scoring(self, mention, history_threshold=0.8):
        # TODO internal convention strength: number of rounds + internal sim score
        # TODO rate internal strength of convention against match with current mention
        mention_embedding = self.scorer.model.encode([mention], convert_to_tensor=True)
        previous_mention_score = {character: [] for character in self.scene_characters}
        for character, history in self.common_ground.history.items():
            if character in self.scene_characters:
                if history:
                    try:
                        # previous_mentions = list(sum(list(zip(history['human'], history['robot']))))
                        previous_mentions = history['human'] + history['robot']
                    except KeyError:
                        previous_mentions = history['human']
                    # logging information
                    logging.debug("History for character: %s", character)
                    for previous in previous_mentions:
                        logging.debug("Previous mention: %s", previous)
                    mention_freq = len(previous_mentions)
                    prev_embeddings = self.scorer.model.encode(previous_mentions, convert_to_tensor=True)
                    strength_scores = util.cos_sim(prev_embeddings, prev_embeddings)
                    strength_avg = torch.mean(strength_scores).item()
                    history_scores = util.cos_sim(mention_embedding, prev_embeddings)
                    recency_weight = torch.tensor()
                    history_scores = history_scores.tolist()
                    history_top = [score for score in history_scores if score >= history_threshold]
                    # history_avg = torch.mean(history_scores).item()
                    # cosine_scores = cosine_scores.tolist()
                    # previous_mention_score[character] = [score[-1] for score in cosine_scores]
                    previous_mention_score[character] = history_score
                    logging.debug("history score for character %s: %s", character, previous_mention_score)

        return previous_mention_score

    def format_response_phrase(self, sex, difference):
        if difference in ['jong', 'oud']:
            phrase = f"die {difference}e {sex}"
        elif difference == 'kaal':
            phrase = f"die kale {sex}"
        elif difference == 'stijl':
            phrase = f"die {sex} met stijl haar"
        elif difference in ['man', 'vrouw', 'jongen']:
            phrase = f"die {difference}"
        elif difference == 'kind':
            phrase = "dat kind"
        else:
            phrase = f"die {sex} met {difference}"

        return phrase

    def save_interaction(self, participant, interaction):
        with open(f'interactions/pp_{participant}_int{interaction}_history.json') as filename:
            json.dump(self.common_ground.history, filename)

    def load_interaction(self, participant, interaction):
        with open(f'interactions/pp_{participant}_int{interaction}_history.json') as filename:
            self.common_ground.history = json.load(filename)



class CommonGround:
    def __init__(self):
        # TODO store history as json at end of each round/end of interaction
        # TODO and load information at start of subsequent interaction
        self.history = {}
        self.positions_discussed = {}
        self.priors = {}
        self.current_position = 1
        self.under_discussion = {'mention': [], 'guess': [], 'position': [], 'response': []}

    def update_priors(self, characters, character_mentioned=None):
        if character_mentioned:
            self.priors[character_mentioned] = 0.1
            self.priors = normalize(self.priors)
        else:
            self.priors = {character: 0.2 for character in characters}

    def add_under_discussion(self, mention, guess=None, position=None, response=None):
        if guess:
            self.under_discussion['guess'].append(guess)
        if position:
            self.under_discussion['position'].append(position)
        self.under_discussion['mention'].append(mention)
        if response:
            self.under_discussion['response'].append(response)

    def reset_under_discussion(self):
        self.under_discussion = {'mention': [], 'guess': [], 'position': [], 'response': []}


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
            # probability_sum = sum(list(listener_probs[word].values()))
            # for referent, probability in listener_probs[word].items():
            #     listener_probs[word][referent] = probability / probability_sum
            listener_probs[word] = normalize(listener_probs[word])

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
            # probability_sum = sum(list(speaker_probs[character].values()))
            # for word, probability in speaker_probs[character].items():
            #     speaker_probs[character][word] = probability / probability_sum
            speaker_probs[character] = normalize(speaker_probs[character])

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
            value_dict[key] = value/value_sum

    return value_dict


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
    ment2 = 'Even denken, nummer 3 was een kale meneer.'
    ment3 = 'Op nummer 5 is dan die bril met wat langer haar.'
    ment4 = "Oké, die eerste is de man met de bril."
    disambiguator = Disambiguator(ak_characters, ak_robot_scene)
    disambiguator.advance_round(start=True)
    # rational = disambiguator.advance_round('3')
    # print(rational)
    disambiguator.advance_round(round_number=6)
    cand, cert, aux = disambiguator.disambiguate(ment2)
    print(cand, cert, aux)
    print(disambiguator.status())
    print(f'O ja, die staat bij mij op plek {aux}')
    disambiguator.advance_round(round_number=3)
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
    # simscorer = SimilarityScorer()
    # item1 = 'deze man heeft een baard'
    # item = 'deze man heeft geen baard'
    # doc1 = nlp(item1)
    # doc2 = nlp(item)
    # sim = doc1.similarity(doc2)
    # print(sim)
    # print(simscorer.calculate_sentence_simscores([item1], [item]))




