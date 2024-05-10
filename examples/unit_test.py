from spot.pragmatic_model.model_ambiguity import Disambiguator, DisambiguatorStatus
from test_world import test_scene, test_phrases, introductions, correct, test_phrases_ambiguity
from spot.pragmatic_model.world_short_phrases_nl import ak_characters
from spot.pragmatic_model.detect_mentions import subtree_right_approach
from datetime import datetime
import logging
import random
from sklearn.metrics import precision_score, recall_score


def test_disambiguator(disambiguator, use_intro=False):
    disambiguator.advance_round(start=True)
    current_pos = 1
    predictions = []
    for j, phrase in enumerate(test_phrases_ambiguity):
        if use_intro:
            if random.random() > 0.5:
                phrase_intro = random.choice(introductions)
                phrase = phrase_intro + ' ' + phrase
        logging.debug("---- ROUND %s -----", disambiguator.current_round)
        logging.debug("--- POSITION %s ----", current_pos)
        logging.debug("Phrase: %s", phrase)
        # mention = subtree_right_approach(phrase)
        # if not mention:
        #     mention = phrase
        selection, certainty, position, response = disambiguator.disambiguate(phrase)
        status = disambiguator.status()
        logging.debug("Disambiguator status: %s", status)
        logging.debug("Selected character %s with certainty %s", selection, certainty)
        logging.debug("Selected response %s", response)
        gold_character = correct[j]
        logging.debug("Correct character: %s", gold_character)
        predictions.append(int(selection))
        if int(selection) == 0:
            logging.debug("Disambigutor failed with no match")
        elif int(selection) == gold_character:
            logging.debug("Disambiguator success")
        else:
            logging.debug("Disambiguator failed with wrong match")
        current_pos += 1
        if current_pos > 4:
            if not j == len(test_phrases)-1:
                disambiguator.advance_round()
                current_pos = 1
        else:
            disambiguator.advance_position()

    return predictions


if __name__ == "__main__":
    logging.basicConfig(filename=f'log_{datetime.now()}.log',
                        level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
    )

    disambiguator = Disambiguator(ak_characters, test_scene, high_engagement=False)

    preds = test_disambiguator(disambiguator, use_intro=True)
    logging.debug("-------RESULTS--------")
    logging.debug("Precision: %s", precision_score(correct, preds, average='micro'))
    logging.debug("Recall: %s", recall_score(correct, preds, average='micro'))


