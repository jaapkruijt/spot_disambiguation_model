import os

import pandas as pd
from spot.pragmatic_model.model_ambiguity import Disambiguator, DisambiguatorStatus
from spot.pragmatic_model.world_short_phrases_nl import ak_characters, ak_robot_scene
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


def evaluate(utterance_data, use_gold_mention=False, filename=None, use_history=True, approach='full'):
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
                selection, certainty, aux = disambiguator.disambiguate(current_utterance, use_history=use_history,
                                                                       approach=approach, literal_threshold=0.6,
                                                                       split_size=2)
                status = disambiguator.status()
                selections.append((selection, certainty, gold, status, current_utterance))
                current_utterance = ''
                disambiguator.advance_position()
        else:
            if row['next_tu'] == 'continue-description':
                current_utterance += text
                continue
            else:
                selection, certainty, aux = disambiguator.disambiguate(text, use_history=use_history,
                                                                       approach=approach, literal_threshold=0.6,
                                                                       split_size=2)
                status = disambiguator.status()
                selections.append((selection, certainty, gold, status, text))
                disambiguator.advance_position()
        # print(disambiguator.common_ground.positions_discussed)
        # print(disambiguator.common_ground.priors)
        # print(disambiguator.common_ground.history)

    for i, selection in enumerate(selections):
        # try:
        #     gold = int(selection[2])
        # except ValueError:
        #     gold = 0
        # if int(selection[0]) == gold:
        #     results['tp'] += 1
        # elif int(selection[0]) == 0:
        #     results['fn'] += 1
        # elif int(selection[0]) != gold:
        #     results['fp'] += 1
        results['pred'].append(int(selection[0]))
        try:
            results['true'].append(int(selection[2]))
        except ValueError:
            results['true'].append(0)
            print(f"No gold for {i}th mention in {filename}")
        results['certainty'].append(selection[1])
        results['status'].append(selection[3])

    results['status'] = Counter(results['status'])

    return results


def evaluate_all_files(directory, average='micro', use_history=True, approach='full', use_gold_mention=True):
    scores = {'precision': {}, 'status': {}, 'recall': {}, 'f1': {}, 'avg_certainty': {}}
    for participant in os.listdir(directory):
        datafile = os.path.join(directory, participant)
        df = read_gold_mentions(datafile)
        result = evaluate(df, use_gold_mention=use_gold_mention, filename=participant, use_history=use_history,
                          approach=approach)
        # precision = result['tp']/(result['tp']+result['fp'])
        # recall = result['tp']/(result['tp']+result['fn'])
        # f1 = 2*((precision*recall)/(precision+recall))
        # scores['f1'].append(f1)
        # scores['tp'].append(result['tp'])
        # scores['fp'].append(result['fp'])
        # scores['fn'].append(result['fn'])
        precision = precision_score(result['true'], result['pred'], average=average, zero_division=0.0)
        recall = recall_score(result['true'], result['pred'], average=average, zero_division=0.0)
        f1 = f1_score(result['true'], result['pred'], average=average)
        avg_certainty = mean(result['certainty'])
        scores['f1'][os.path.splitext(participant)[0]] = f1
        scores['avg_certainty'][os.path.splitext(participant)[0]] = avg_certainty
        scores['precision'][os.path.splitext(participant)[0]] = precision
        scores['recall'][os.path.splitext(participant)[0]] = recall
        scores['status'][os.path.splitext(participant)[0]] = result['status']

    return scores




if __name__ == '__main__':
    # dataframe = read_gold_mentions('/Users/jaapkruijt/Documents/GitHub/spot_disambiguation_model/local_files/test_files/AK_41.tsv')
    # res = evaluate(dataframe, use_gold_mention=False)
    # print(res)
    # print(precision_score(res['true'], res['pred'], average='micro'))
    # print(recall_score(res['true'], res['pred'], average='micro'))
    filepath = '/Users/jaapkruijt/Documents/GitHub/spot_disambiguation_model/local_files/test_files'
    all_results = []
    full_micro = evaluate_all_files(filepath, use_gold_mention=False)
    participants = list(full_micro['precision'].keys())
    participants_doubled = list(zip(participants, participants))
    participants_new = list(sum(participants_doubled, ()))
    precision_recall = ['precision', 'recall']*10
    tuples = list(zip(participants_new, precision_recall))
    index = pd.MultiIndex.from_tuples(tuples)
    configs = [{'avg': 'micro', 'history': True, 'approach': 'full'}, {'avg': 'macro', 'history': True, 'approach': 'full'},
               {'avg': 'micro', 'history': True, 'approach': 'literal'}, {'avg': 'macro', 'history': True, 'approach': 'literal'},
               {'avg': 'micro', 'history': False, 'approach': 'full'}, {'avg': 'macro', 'history': False, 'approach': 'full'},
               {'avg': 'micro', 'history': False, 'approach': 'literal'}, {'avg': 'macro', 'history': False, 'approach': 'literal'}]

    for config in configs:
        results = evaluate_all_files(filepath, use_gold_mention=False, average=config['avg'],
                                     use_history=config['history'], approach=config['approach'])
        prec_rec = list(sum(list(zip(results['precision'].values(), results['recall'].values())), ()))
        all_results.append(prec_rec)



