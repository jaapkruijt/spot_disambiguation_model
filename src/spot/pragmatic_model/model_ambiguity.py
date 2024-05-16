from pathlib import Path

import math
from enum import Enum
import logging
from scipy.stats import entropy
from scipy.special import rel_entr
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import random
from random import choice

import torch
from sentence_transformers import SentenceTransformer, util
from spot.pragmatic_model.world_short_phrases_nl import ak_characters, ak_robot_scene
from gensim.models import word2vec, KeyedVectors
import numpy as np
import json
import os
from difflib import SequenceMatcher
import re

from tqdm import tqdm
from string import punctuation
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
    '''
    Output status of the disambiguator after disambigation. Determines the next action for the dialog manager
    '''
    AWAIT_NEXT = 1
    SUCCESS_HIGH = 2
    SUCCESS_LOW = 3
    NO_MATCH = 4
    MATCH_MULTIPLE = 5
    MATCH_PREVIOUS = 6
    NEG_RESPONSE = 7


class Disambiguator:
    def __init__(self, world, scenes, high_engagement=bool, force_commit: bool = True):
        '''
        Initialize the disambiguator with a closed game world consisting of scenes and positions of characters in the
        scenes, and visual information about the characters. Sets the current round to 0, and initializes the Common
        Ground, Lexicon and Similarity Scorer associated with the disambiguator.
        :param world: dict. Contains visual information about the characters
        :param scenes: dict. Contains a dict of characters and their positions per round.
        :param high_engagement: Bool. If True, the robot engages in convention formation by using mentions. If False,
        the robot does not use mentions other than 'die'/'that one'
        '''
        self._status = DisambiguatorStatus.AWAIT_NEXT
        self._uncommitted_status = None
        self.common_ground = CommonGround()
        self.lexicon = Lexicon()
        self.scorer = SimilarityScorer()
        self.world = world
        self.scenes = scenes
        self.scene_characters = []
        self.current_round = 0
        self.high_engagement = high_engagement
        self.vectorizer = TfidfVectorizer()
        self.nlp = spacy.load('nl_core_news_lg')

        self._force_commit = force_commit

    def status(self):
        '''
        Returns the name of the current DisambiguatorStatus
        :return: str
        '''
        return self._status.name

    def advance_round(self, round_number=None, start=False):
        '''
        Called to move the disambiguator to the next round in the game. Updates the round number, scene characters,
        pragmatic lexicon, status and common ground information
        :param round_number: Optional. If a round number is supplied, the disambiguator moves to that round.
        If not supplied the disambiguator moves up one round from the current round
        :param start: Bool. If True, the base lexicon is additionally updated (only happens once at the start). If
        False, the information for the last character discussed is added to the common ground.
        :return:
        '''
        if start:
            self.lexicon.compute_base_lexicon(list(self.world.values()))
        else:
            self.confirm_character_position()
            self.find_preferred_conventions()
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

        # TODO should not return anything, check!
        return self.lexicon.pragmatic_lexicon()

    def advance_position(self, position=None, skip=False):
        '''
        Called to move the disambiguator to the next position in a round. Updates the disambiguator status and
        adds information for the last character discussed to the common ground
        :param position: Optional. If a position is supplied, the disambiguator moves to that position in the round.
        If not supplied, moves up one position from the current.
        :return: NA
        '''
        if not skip:
            self.confirm_character_position()
        self._status = DisambiguatorStatus.AWAIT_NEXT
        if position:
            self.common_ground.current_position = position
        else:
            self.common_ground.current_position += 1
        self.common_ground.reset_under_discussion()

    def confirm_character_position(self):
        '''
        After (optional) confirmation from the disambiguator, adds the following information for the selected character
        to the common ground: selected character id, mention used by the human, mention used by the robot (only if
        high_engagement is True), the selected character's position in the human view. Also updates the prior
        probability for the selected character to reflect that they have already been discussed this round.

        :return: NA
        '''
        selection = self.common_ground.under_discussion['guess'][-1]
        mention = self.common_ground.under_discussion['mention'][-1]
        logging.debug('Selection: %s', selection)
        logging.debug('Mention added to history: %s', mention)
        if selection in self.common_ground.history.keys():
            self.common_ground.history[selection]['human'].append(mention)
        else:
            self.common_ground.history[selection] = {'human': [mention]}
        if self.high_engagement:
            response = self.common_ground.under_discussion['response'][-1]
            if response:
                if 'robot' in self.common_ground.history[selection].keys():
                    self.common_ground.history[selection]['robot'].append(response)
                else:
                    self.common_ground.history[selection]['robot'] = [response]
        self.common_ground.positions_discussed[str(self.current_round)][self.common_ground.current_position] = selection
        self.common_ground.update_priors(self.scene_characters, character_mentioned=selection)

    def commit_status(self):
        self._status = self._uncommitted_status[0]
        self.common_ground.add_under_discussion(*self._uncommitted_status[1:])
        self._uncommitted_status = None

    def disambiguate(self, mention, approach='full', history_factor=1.0, test=False, literal_threshold=0.7,
                     history_threshold=0.4, split_size=2, certainty_threshold=0.60, force_commit=True):
        """
        Main function of the disambiguator used to identify a character based on a description. First checks its status
        to determine whether repair is necessary. Combines a literal cosine-similarity-based score with a score based on
        information from the history of mentions to determine the most likely referent. Its next action and result
        depend on the distribution of the likelihood scores.
        :param mention: string
        :param approach: 'full' or 'literal'
        :param use_history: bool
        :param test: bool
        :param literal_threshold: float
        :param history_threshold: float
        :param split_size: int
        :param certainty_threshold: float
        :return: selection, certainty, position, difference, await continuation
        """
        # Strip the mention of any leading or trailing spaces and punctuation
        mention.strip()
        mention.strip(punctuation)
        logging.debug("Mention: %s", mention)
        logging.debug("Status: %s", self.status())

        # check if coming from repair, positive response or new input
        if self.status() == 'MATCH_MULTIPLE':
            if re.search(r"\bja\b", mention.lower()):
                self._status = DisambiguatorStatus.SUCCESS_HIGH
                selected = self.common_ground.under_discussion['guess'][-1]
                position = self.common_ground.under_discussion['position'][-1]
                response = self.common_ground.under_discussion['response'][-1]

                return selected, 1.0, int(position), response, False
            # TODO duplicate code, see line 268
            elif re.search(r"\bnee\b", mention.lower()):
                self._status = DisambiguatorStatus.NEG_RESPONSE
                return '0', 1.0, None, None, False

        # compute similarity scores with history
        history_score = self.mention_history_scoring(mention, history_threshold)

        # compute similarity scores with attributes
        literal_candidate_attributes, literal_candidate_scores = self.literal_match(mention, threshold=literal_threshold, split_size=split_size)
        for character, attributes in literal_candidate_attributes.items():
            logging.debug("Attributes for character %s: %s", character, attributes)
        # logging.debug("Scores before history: %s", literal_candidate_scores)

        # combine literal score and history score through weighted averaging
        for character, (score, weight) in history_score.items():
            literal_candidate_scores[character] = np.average(np.array([literal_candidate_scores[character], score]),
                                                             weights=np.array([1.0, weight*history_factor]))
        # logging.debug("Scores after history scoring: %s", literal_candidate_scores)

        # update the prior probability for each character
        for character, score in literal_candidate_scores.items():
            literal_candidate_scores[character] *= self.common_ground.priors[character]
        # TODO can we do it with lists/arrays like below?
        # literal_scores = [literal_candidate_scores[character]*self.common_ground.priors[character]
        #                   for character in literal_candidate_scores.keys()]
        # logging.debug("Scores after prior: %s", literal_candidate_scores)

        # TODO
        try:
            # lower probability for original guess in case of negative feedback
            if self.status() in ['MATCH_PREVIOUS', 'SUCCESS_LOW', 'NEG_RESPONSE']:
                # TODO check if too thorough
                for previous_guess in self.common_ground.under_discussion['guess']:
                    literal_candidate_scores[previous_guess] -= 0.1
                    if literal_candidate_scores[previous_guess] < 0.0:
                        literal_candidate_scores[previous_guess] = 0.0
        except Exception as e:
            logging.exception("Failed to lower probability")

        # normalize scores to get distribution summing to one
        literal_candidate_scores = normalize(literal_candidate_scores)
        # scores = np.array(list(literal_candidate_scores.values()))
        # scores = sorted(list(literal_candidate_scores.values()))
        sorted_candidates_scores = dict(sorted(literal_candidate_scores.items(), key=lambda item: item[1], reverse=True))
        sorted_candidates = list(sorted_candidates_scores.keys())
        sorted_scores = list(sorted_candidates_scores.values())
        logging.debug("Score distribution: %s", sorted_scores)
        # max_score = scores.max(initial=0)
        max_score = list(sorted_scores)[0]
        selected = '0'

        # In case no match was found
        if max_score == 0.0:
            # TODO duplicate code, see line 218
            if re.search(r"\bnee\b", mention.lower()):
                self._status = DisambiguatorStatus.NEG_RESPONSE
                return '0', 1.0, None, None, False
            if force_commit or self._force_commit:
                self._status = DisambiguatorStatus.NO_MATCH
                self.common_ground.add_under_discussion(mention, selected)
                return selected, 1.0, None, None, False
            else:
                self._uncommitted_status = (DisambiguatorStatus.NO_MATCH, mention, selected, None, None)
                return selected, 1.0, None, None, True

        # find top candidate and compute certainty
        uniform = [1/len(sorted_scores)]*len(sorted_scores)
        # logging.debug("Uniform distribution: %s", uniform)
        certainty = sum(rel_entr(sorted_scores, uniform))/math.log(len(sorted_scores))
        logging.debug("Certainty according to KL: %s", certainty)
        top_candidates = []
        # for candidate, score in literal_candidate_scores.items():
        #     if score == max_score:
        #         top_candidates.append(candidate)
        for i, candidate in enumerate(sorted_candidates):
            if i == 0:
                top_candidates.append(candidate)
                logging.debug("Added character %s to candidates", candidate)
            elif i == len(sorted_candidates)-1:
                continue
            else:
                if sorted_candidates_scores[sorted_candidates[i-1]] - sorted_candidates_scores[candidate] > \
                        (sorted_candidates_scores[candidate] - sorted_candidates_scores[sorted_candidates[i+1]])/2:
                    break
                else:
                    logging.debug("Added character %s to candidates", candidate)
                    top_candidates.append(candidate)

        # case: one top candidate
        if len(top_candidates) == 1:
            selected = top_candidates[0]
            if selected in self.common_ground.positions_discussed[str(self.current_round)].values():
                self._status = DisambiguatorStatus.MATCH_PREVIOUS

                position = self.scenes[str(self.current_round)][selected]
                self.common_ground.add_under_discussion(mention, selected, position)
                return selected, certainty, int(position), None, False
            else:
                if certainty > certainty_threshold:
                    self._status = DisambiguatorStatus.SUCCESS_HIGH
                    position = self.scenes[str(self.current_round)][selected]

                    if selected in self.common_ground.preferred_convention:
                        response = self.common_ground.preferred_convention[selected]
                    else:
                        try:
                            response = self.format_response_phrase(random.choice(self.world[selected]['gender']),
                                                               random.choice(list(literal_candidate_attributes[selected])))
                        except IndexError:
                            response = 'die'
                    self.common_ground.add_under_discussion(mention, selected, position, response)
                    return selected, certainty, int(position), response, False
                else:
                    self._status = DisambiguatorStatus.SUCCESS_LOW
                    if selected in self.common_ground.preferred_convention:
                        response = self.common_ground.preferred_convention[selected]
                    else:
                        response, selected = self.find_and_select_differences(self.scene_characters,
                                                                              single_candidate=selected)
                    position = self.scenes[str(self.current_round)][selected]
                    self.common_ground.add_under_discussion(mention, selected, position, response)
                    return selected, certainty, int(position), response, False

        # case: more than one top candidate
        if len(top_candidates) > 1:
            available_candidates = []
            for candidate in top_candidates:
                if candidate not in self.common_ground.positions_discussed[str(self.current_round)].values():
                    available_candidates.append(candidate)
            if not available_candidates:
                self._status = DisambiguatorStatus.MATCH_PREVIOUS
                selected = top_candidates[0]
                position = self.scenes[str(self.current_round)][selected]
                self.common_ground.add_under_discussion(mention, selected, position)
                return selected, certainty, int(position), None, False
            elif len(available_candidates) == 1:
                selected = available_candidates[0]
                self._status = DisambiguatorStatus.SUCCESS_LOW
                if selected in self.common_ground.preferred_convention:
                    response = self.common_ground.preferred_convention[selected]
                else:
                    response, selected = self.find_and_select_differences(top_candidates,
                                                                          single_candidate=selected)
                position = self.scenes[str(self.current_round)][selected]
                self.common_ground.add_under_discussion(mention, selected, position, response)
                return selected, certainty, int(position), response, False
            else:
                if force_commit or self._force_commit:
                    self._status = DisambiguatorStatus.MATCH_MULTIPLE
                if approach == 'full':
                    top_candidate_attributes = {candidate: attributes for candidate, attributes
                                                in literal_candidate_attributes.items() if candidate in top_candidates}
                    pragmatic_match = self.contextual_pragmatic_match(top_candidate_attributes)
                    pragmatic_match = normalize(pragmatic_match)
                    ordered = dict(sorted(pragmatic_match.items(), key=lambda item: item[1], reverse=True))
                    # logging.debug("Ordered dict: %s", ordered)
                    selected = next(iter(ordered))
                    # logging.debug('Selected character: %s', selected)
                    response, selected = self.find_and_select_differences(top_candidates, single_candidate=selected)
                    scores = np.array(list(ordered.values()))
                    uniform = [1/len(scores)]*len(scores)
                    # logging.debug("Uniform distribution: %s", uniform)
                    certainty = sum(rel_entr(scores, uniform))/math.log(len(scores))
                    # score_entropy = entropy(scores, base=2)
                    # certainty = 1-(score_entropy/math.log(len(ordered), 2))
                    position = self.scenes[str(self.current_round)][selected]
                else:
                    response, selected = self.find_and_select_differences(top_candidates)
                    position = self.scenes[str(self.current_round)][selected]

                if force_commit or self._force_commit:
                    self.common_ground.add_under_discussion(mention, selected, position, response)
                    return selected, certainty, int(position), response, False
                else:
                    self._uncommitted_status = (DisambiguatorStatus.MATCH_MULTIPLE, mention, selected, position, response)
                    return selected, certainty, int(position), response, True

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
                if isinstance(difference, list):
                    difference = random.choice(difference)
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
                if isinstance(difference, list):
                    difference = random.choice(difference)
                logging.debug("No differences found")

        phrase = self.format_response_phrase(random.choice(self.world[candidate_guess]['gender']), difference)

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
        # for part in mention_parts:
            # logging.debug("Mention part: %s", part)
        # score mention parts against attributes
        scoring = self.scorer.calculate_sentence_simscores(mention_parts, list(self.lexicon.base_lexicon().keys()))
        scoring = scoring.tolist()
        matches = []
        for section in scoring:
            matches.append(list(zip(section, list(self.lexicon.base_lexicon().keys()))))
        # mention_text = {}
        for i, match in enumerate(matches):
            losing_lengths = self.find_losing_hair_lengths(match)
            losing_colours = self.find_losing_hair_colours(match)
            for score, attribute in match:
                if attribute not in losing_lengths and attribute not in losing_colours:
                    if score >= threshold:
                        # logging.debug("Match for mention part %s with %s, score: %s", mention_parts[i], attribute, score)
                        # mention_text[mention_parts[i]] = None
                        for character, value in self.lexicon.base_lexicon()[attribute].items():
                            if value == 1 and character in self.scene_characters:
                                # avoid multiple mentions
                                if attribute not in candidate_attributes[character]:
                                    logging.debug("Found match for attribute %s for character %s", attribute, character)
                                    candidate_attributes[character].add(attribute)
                                    candidate_scores[character] += score

        candidate_scores = normalize(candidate_scores)

        # mention_string = ''
        # for i, part in enumerate(mention_text):
        #     if i == 0:
        #         mention_string += part
        #     else:
        #         mention_string += ' ' + part.split()[-1]

        return candidate_attributes, candidate_scores

    def find_losing_hair_colours(self, match):
        colours = {}
        hair_colours = ['bruin haar', 'grijs haar', 'zwart haar', 'blond haar', 'donkerblond haar', 'donker haar']
        for score, attribute in match:
            if attribute in hair_colours:
                colours[attribute] = score
        sorted_colours = dict(sorted(colours.items(), key=lambda item: item[1], reverse=True))
        winner = next(iter(sorted_colours))
        losers = list(colours.keys())
        losers.remove(winner)

        return losers

    def find_losing_hair_lengths(self, match):
        lengths = {}
        hair_lengths = ['kort haar', 'lang haar', 'halflang haar']
        for score, attribute in match:
            if attribute in hair_lengths:
                lengths[attribute] = score
        sorted_lengths = dict(sorted(lengths.items(), key=lambda item: item[1], reverse=True))
        winner = next(iter(sorted_lengths))
        losers = list(lengths.keys())
        losers.remove(winner)

        return losers

    def contextual_pragmatic_match(self, candidates):
        total = {candidate: 0 for candidate in candidates.keys()}
        for candidate, attributes in candidates.items():
            for attribute in attributes:
                probability = self.lexicon.pragmatic_lexicon()[attribute][candidate]
                total[candidate] += probability

        return total


    def mention_history_scoring(self, mention, history_threshold=0.4):
        # TODO internal convention strength: number of rounds + internal sim score
        # TODO rate internal strength of convention against match with current mention
        mention_embedding = self.scorer.model.encode([mention], convert_to_tensor=True)
        previous_mention_score = {character: (0.0, 0.0) for character in self.scene_characters}
        for character, history in self.common_ground.history.items():
            if character in self.scene_characters:
                if history:
                    try:
                        previous_mentions = list(sum(list(zip(history['human'], history['robot'])), ()))
                        # previous_mentions = history['human'] + history['robot']
                    except KeyError:
                        previous_mentions = history['human']
                    # logging information
                    logging.debug("History for character: %s", character)
                    # for previous in previous_mentions:
                        # logging.debug("Previous mention: %s", previous)
                    mention_freq = len(previous_mentions)
                    prev_embeddings = self.scorer.model.encode(previous_mentions, convert_to_tensor=True)
                    strength_scores = util.cos_sim(prev_embeddings, prev_embeddings)
                    # TODO does it make sense that strength is high when only one mention in history?
                    strength_avg = torch.mean(torch.flatten(strength_scores)).item()
                    logging.debug("Average history strength: %s", strength_avg)
                    cosine_scores = util.cos_sim(mention_embedding, prev_embeddings)
                    cosine_scores = torch.flatten(cosine_scores)
                    # logging.debug("Flattened cosine scores: %s", cosine_scores)
                    # cosine_scores = torch.mul(cosine_scores, strength_avg)
                    cosine_scores = cosine_scores.to('cpu')
                    # logging.debug("Cosine scores mutiplied by weight: %s", cosine_scores)
                    seq_matches = []
                    seq = SequenceMatcher()
                    seq.set_seq2(mention)
                    for previous in previous_mentions:
                        seq.set_seq1(previous)
                        match = seq.find_longest_match()
                        # logging.debug("Longest match info for history %s: %s", previous, match)
                        match_ratio = seq.ratio()
                        # logging.debug("Match ratio for history %s: %s", previous, match_ratio)
                        seq_matches.append(match_ratio)
                    seq_scores = torch.tensor(seq_matches).to('cpu')
                    history_scores = cosine_scores*seq_scores
                    # logging.debug("Combined score: %s", history_scores)
                    recency_weights = torch.tensor([1/i for i in range(len(history_scores), 0, -1)])
                    # logging.debug("Recency for history: %s", recency_weights)
                    history_scores = history_scores*recency_weights
                    # logging.debug("Combined scores weighed by recency: %s", history_scores)
                    # history_scores = torch.nn.functional.normalize(history_scores, dim=0)
                    history_scores = history_scores.tolist()
                    history_score = sum(history_scores)
                    if history_score > history_threshold:
                        previous_mention_score[character] = (history_score, strength_avg)
        logging.debug("Avg history score: %s", previous_mention_score)

        return previous_mention_score

    def find_preferred_conventions(self, threshold=0.4):
        mention_corpus = []
        mention_characters = []
        for character, history in self.common_ground.history.items():
            if len(history['human']) >= 3:
                mention_corpus.append('. '.join(history['human']))
                mention_characters.append(character)

        if not mention_corpus:
            return

        for character, history in self.common_ground.history.items():
            if character in mention_characters:
                continue
            else:
                mention_corpus.append('. '.join(history['human']))

        convention_words = {character: [] for character in mention_characters}
        tfidfvectors = self.vectorizer.fit_transform(mention_corpus)
        words = self.vectorizer.get_feature_names_out()
        for i, character in enumerate(mention_characters):
            for j, tfidfscore in enumerate(tfidfvectors[i].toarray()[0]):
                if tfidfscore > threshold:
                    convention_words[character].append(words[j])

        for character in tqdm(mention_characters):
            logging.debug("Finding preferred convention for character %s", character)
            if convention_words[character]:
                logging.debug('Salient words: %s', convention_words[character])
                recent_utterance = self.common_ground.history[character]['human'][-1]
                doc = self.nlp(recent_utterance)
                relative_head = {'relative_head': None}
                structure = {'amod': '', 'head': '', 'nmod': ''}
                for token in doc:
                    if token.text in convention_words[character]:
                        if token.pos_ not in ['DET', 'AUX']:
                            if token.head.text not in convention_words[character]:
                                relative_head['relative_head'] = token
                if relative_head['relative_head']:
                    structure['head'] = relative_head['relative_head'].text
                    logging.debug("Head of convention: %s", structure['head'])
                    for child in relative_head['relative_head'].children:
                        if child.dep_ in ['amod', 'nmod']:
                            subtree_span = doc[child.left_edge.i: child.right_edge.i+1]
                            structure[child.dep_] = subtree_span.text
                    if not structure['amod'] and not structure['nmod']:
                        logging.debug("No modifiers found from head")
                        subtree_span = doc[relative_head['relative_head'].left_edge.i: relative_head['relative_head'].right_edge.i+1]
                        structure['head'] = subtree_span.text
                    self.common_ground.preferred_convention[character] = 'die ' + ' '.join([value for value in list(structure.values()) if value])

    def format_response_phrase(self, sex, difference):
        if difference in ['jong', 'oud']:
            if sex in ['jongetje', 'meisje']:
                phrase = f"dat {difference}e {sex}"
            else:
                phrase = f"die {difference}e {sex}"
        elif difference == 'kaal':
            phrase = f"die kale {sex}"
        elif difference == 'stijl':
            phrase = f"die {sex} met stijl haar"
        elif difference in ['man', 'vrouw', 'jongen']:
            phrase = f"die {difference}"
        elif difference in ['kind', 'jongetje', 'meisje']:
            phrase = f"dat {difference}"
        else:
            phrase = f"die {sex} met {difference}"

        return phrase

    def save_interaction(self, storage_dir, participant, interaction):
        path = f'{storage_dir}/conventions'
        Path(path).mkdir(parents=True, exist_ok=True)

        common_ground = {'history': self.common_ground.history, 'conventions': self.common_ground.preferred_convention}

        with open(os.path.join(path, f'pp_{participant}_int{interaction}_history.json'), 'w') as fname:
            json.dump(common_ground, fname)

    def load_interaction(self, storage_dir, participant, interaction):
        path = f'{storage_dir}/conventions'
        with open(os.path.join(path, f'pp_{participant}_int{interaction}_history.json'), 'w') as filename:
            common_ground = json.load(filename)
            self.common_ground.history = common_ground['history']
            self.common_ground.preferred_convention = common_ground['conventions']


class CommonGround:
    def __init__(self):
        # TODO store history and preferred convention as json at end of each round/end of interaction
        # TODO and load information at start of subsequent interaction
        self.history = {}
        self.positions_discussed = {}
        self.priors = {}
        self.current_position = 1
        self.under_discussion = {'mention': [], 'guess': [], 'position': [], 'response': []}
        self.preferred_convention = {}

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
                if type(attribute) is list:
                    for attr in attribute:
                        attributes[attr] = {}
                else:
                    attributes[attribute] = {}

        return attributes

    def compute_base_lexicon(self, characters):
        """
        :param characters: list
        :return:
        """
        lexicon = self.get_attributes(characters)

        for i, character in enumerate(characters):
            sub_values = [sub_value for value in character.values() for sub_value in value if
                          isinstance(value, list)]
            character_attributes = [value for value in character.values() if not isinstance(value, list)]
            character_attributes.extend(sub_values)
            for attribute in lexicon:
                if attribute in character_attributes:
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
    import pandas as pd

    corpus = ['Dat is een man met een baard', 'Dit is de vrouw met de oorbellen', 'Dat is een kale man',
              'Dat is brilmans']
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(corpus)
    names = tfidf.get_feature_names_out()
    print(names)
    print(vectors.shape)
    print(vectors)
    print(tfidf.get_params())
    first = vectors[0]
    df = pd.DataFrame(first.T.todense(), index=tfidf.get_feature_names_out(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    print(df)




