from spot.pragmatic_model.model_ambiguity import Disambiguator, DisambiguatorStatus
from test_world import test_scene, test_phrases, introductions, correct
from spot.pragmatic_model.world_short_phrases_nl import ak_characters
from datetime import datetime
import logging
import random


def test_disambiguator(disambiguator, use_intro=False):
    disambiguator.advance_round(start=True)
    current_pos = 1
    for j, phrase in enumerate(test_phrases):
        if use_intro:
            phrase_intro = random.choice(introductions)
            phrase = phrase_intro + ' ' + phrase
        logging.debug("---- ROUND %s -----", disambiguator.current_round)
        logging.debug("--- POSITION %s ----", current_pos)
        selection, certainty, position, difference = disambiguator.disambiguate(phrase)
        status = disambiguator.status()
        logging.debug("Disambiguator status: %s", status)
        logging.debug("Selected character %s with certainty %s", selection, certainty)
        logging.debug("Selected difference %s as attribute", difference)
        gold_character = correct[j]
        logging.debug("Correct character: %s", gold_character)
        if int(selection) == 0:
            logging.debug("Disambigutor failed with no match")
        elif int(selection) == gold_character:
            logging.debug("Disambiguator success")
        else:
            logging.debug("Disambiguator failed with wrong match")
        # TODO need to get:
        # distribution of scores
        # attribute matches
        # history score per character
        # list of differences
        disambiguator.advance_position()
        current_pos += 1
        if current_pos > 4:
            if not j == len(test_phrases)-1:
                disambiguator.advance_round()
                current_pos = 1


if __name__ == "__main__":
    logging.basicConfig(filename=f'log_{datetime.now()}.log',
                        level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
    )

    disambiguator = Disambiguator(ak_characters, test_scene)

    test_disambiguator(disambiguator)

