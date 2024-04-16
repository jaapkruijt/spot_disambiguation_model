import spacy
import re
from thefuzz import fuzz
from spacy import displacy

nlp = spacy.load('nl_core_news_lg')


def subtree_right_approach(utterance):
    mention_list = []
    optional_left_list = []
    doc = nlp(utterance)
    for token in doc:
        subject_mentioned = False
        if token.head.text == token.text:
            if token.pos_ == 'NOUN':
                if token.text.lower() != 'ik':
                    # mention_list.append(token.text)
                    subject_mentioned = True
                    for t in token.subtree:
                        mention_list.append(t.text)
            else:
                for child in token.rights:
                    if child.dep_ == 'nsubj':
                        subject_mentioned = True
                    if not subject_mentioned:
                        continue
                    else:
                        for t in child.subtree:
                            mention_list.append(t.text)

    mention = ' '.join(mention_list)

    if optional_left_list:
        lefts = ' '.join(optional_left_list)
        mention = lefts + ' ' + mention

    return mention


def nsubj_approach(utterance):
    mention_list = []
    doc = nlp(utterance)
    for token in doc:
        if token.dep_ == 'nsubj':
            subtree = [t for t in token.subtree]
            for t in subtree:
                mention_list.append(t.text)

    mention = ' '.join(mention_list)

    return mention


if __name__ == "__main__":
    mention = 'ik heb een vrouw met oorbellen'
    result = subtree_right_approach(mention)
    result2 = nsubj_approach(mention)
    print(result)
    print(result2)
    print(spacy.explain('nsubj'))
    doc = nlp(mention)
    svg = displacy.render(doc, style="dep")
    output_path = "spacy_tree_structure.svg"
    with open(output_path, 'w') as output:
        output.write(svg)
    # doc = nlp("En dan staat bij mij op nummer 5 een meneer met het grijze baardje.")
    # doc = nlp("Dan staat er een mevrouw met een beetje het haar wat boven de schouders.")
