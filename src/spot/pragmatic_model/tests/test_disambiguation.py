import os

import pandas as pd
from src.spot.pragmatic_model.model_ambiguity import Disambiguator, DisambiguatorStatus
from src.spot.pragmatic_model.world_short_phrases_nl import ak_characters, ak_robot_scene
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from tabulate import tabulate
from collections import Counter
from statistics import mean

def read_gold_mentions(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df = df.loc[(df['speaker'] == 'human') & (df['mention'].notna()) & (df['round'] != 0) &
                (df['tu_relation'] != 'summarization')]
    df = df.reset_index()
    df['previous'] = df['round'].shift(1)
    df['next_tu'] = df['tu_relation'].shift(-1)

    return df


def evaluate(utterance_data, use_gold_mention=False):
    selections = []
    # results = {'tp': 0, 'fp': 0, 'fn': 0, 'certainty': [], 'status': []}
    results = {'pred': [], 'true': [], 'certainty': [], 'status': []}
    disambiguator = Disambiguator(ak_characters, ak_robot_scene)
    disambiguator.advance_round(start=True)
    current_utterance = ''
    for index, row in utterance_data.iterrows():
        if not math.isnan(row['previous']) and row['previous'] != row['round']:
            disambiguator.advance_round()
        text = row['mention'] if use_gold_mention else row['text']
        gold = row['character']
        if current_utterance:
            if row['next_tu'] == 'continue-description':
                current_utterance += text
                continue
            else:
                selection, certainty, aux = disambiguator.disambiguate(current_utterance, literal_threshold=0.6, split_size=2)
                status = disambiguator.status()
                selections.append((selection, certainty, gold, status, current_utterance))
                current_utterance = ''
                disambiguator.advance_position()
        else:
            if row['next_tu'] == 'continue-description':
                current_utterance += text
                continue
            else:
                selection, certainty, aux = disambiguator.disambiguate(text, literal_threshold=0.6, split_size=2)
                status = disambiguator.status()
                selections.append((selection, certainty, gold, status, text))
                disambiguator.advance_position()
        # print(disambiguator.common_ground.positions_discussed)
        # print(disambiguator.common_ground.priors)
        # print(disambiguator.common_ground.history)

    for selection in selections:
        # if int(selection[0]) == 0:
        #     results['fn'] += 1
        # elif int(selection[0]) == int(selection[2]):
        #     results['tp'] += 1
        # elif int(selection[0]) != int(selection[2]):
        #     results['fp'] += 1
        results['pred'].append(int(selection[0]))
        try:
            results['true'].append(int(selection[2]))
        except ValueError:
            results['true'].append(0)
        results['certainty'].append(selection[1])
        results['status'].append(selection[3])

    results['status'] = Counter(results['status'])

    return results


def evaluate_all_files(directory):
    scores = {'precision': [], 'status': Counter(), 'recall': [], 'f1': [], 'avg_certainty': []}
    for participant in os.listdir(directory):
        datafile = os.path.join(directory, participant)
        df = read_gold_mentions(datafile)
        result = evaluate(df, use_gold_mention=True)
        # precision = result['tp']/(result['tp']+result['fp'])
        # recall = result['tp']/(result['tp']+result['fn'])
        # f1 = 2*((precision*recall)/(precision+recall))
        # scores['f1'].append(f1)
        # scores['tp'].append(result['tp'])
        # scores['fp'].append(result['fp'])
        # scores['fn'].append(result['fn'])
        precision = precision_score(result['true'], result['pred'], average='micro')
        recall = recall_score(result['true'], result['pred'], average='micro')
        f1 = f1_score(result['true'], result['pred'], average='micro')
        avg_certainty = mean(result['certainty'])
        scores['f1'].append(f1)
        scores['avg_certainty'].append(avg_certainty)
        scores['precision'].append(precision)
        scores['recall'].append(recall)
        scores['status'].update(result['status'])

    return scores




if __name__ == '__main__':
    # dataframe = read_gold_mentions('/Users/jaapkruijt/Documents/GitHub/spot_disambiguation_model/local_files/test_files/AK_41.tsv')
    # res = evaluate(dataframe, use_gold_mention=False)
    # print(res)
    # print(precision_score(res['true'], res['pred'], average='micro'))
    # print(recall_score(res['true'], res['pred'], average='micro'))
    total = evaluate_all_files('/Users/jaapkruijt/Documents/GitHub/spot_disambiguation_model/local_files/test_files')
    print(total)