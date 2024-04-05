import os

import pandas as pd
from spot.pragmatic_model.model_ambiguity import Disambiguator, DisambiguatorStatus
from spot.pragmatic_model.world_short_phrases_nl import ak_characters, ak_robot_scene
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from tabulate import tabulate
from collections import Counter
from statistics import mean
from spot.pragmatic_model.detect_mentions import subtree_right_approach

def read_gold_mentions(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df = df.loc[(df['speaker'] == 'human') & (df['mention'].notna()) & (df['round'] != 0) &
                (df['tu_relation'] != 'summarization')]
    df = df.reset_index()
    df['next'] = df['round'].shift(-1)
    df['next_tu'] = df['tu_relation'].shift(-1)

    return df


def evaluate(utterance_data, use_gold_mention=False, filename=None, history_factor=1.0, approach='full', mention_det=False):
    selections = []
    # results = {'tp': 0, 'fp': 0, 'fn': 0, 'certainty': [], 'status': []}
    results = {'pred': [], 'true': [], 'certainty': [], 'status': []}
    disambiguator = Disambiguator(ak_characters, ak_robot_scene, high_engagement=False)
    disambiguator.advance_round(start=True)
    current_utterance = ''
    for index, row in utterance_data.iterrows():
        text = row['mention'] if use_gold_mention else row['text']
        gold = row['character']
        if current_utterance:
            if row['next_tu'] == 'continue-description':
                current_utterance += text
                continue
            else:
                if mention_det:
                    mention = subtree_right_approach(current_utterance)
                    if not mention:
                        mention = current_utterance
                else:
                    mention = current_utterance
                selection, certainty, position, response = disambiguator.disambiguate(mention, history_factor=history_factor,
                                                                       approach=approach, literal_threshold=0.7,
                                                                       split_size=2)
                status = disambiguator.status()
                selections.append((selection, certainty, gold, status, mention))
                current_utterance = ''
                if not math.isnan(row['next']) and row['next'] != row['round']:
                    disambiguator.advance_round()
                else:
                    disambiguator.advance_position()
        else:
            if row['next_tu'] == 'continue-description':
                current_utterance += text
                continue
            else:
                if mention_det:
                    mention = subtree_right_approach(text)
                    if not mention:
                        mention = text
                else:
                    mention = text
                selection, certainty, position, difference = disambiguator.disambiguate(mention, history_factor=history_factor,
                                                                       approach=approach, literal_threshold=0.7,
                                                                       split_size=2)
                status = disambiguator.status()
                selections.append((selection, certainty, gold, status, mention))
                if not math.isnan(row['next']) and row['next'] != row['round']:
                    disambiguator.advance_round()
                else:
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


def evaluate_all_files(participant_list, path, average='micro', history_factor=1.0, approach='full',
                       use_gold_mention=True, mention_det=False):
    scores = {'precision': {}, 'status': {}, 'recall': {}, 'f1': {}, 'avg_certainty': {}}
    for participant in participant_list:
        datafile = os.path.join(path, participant)
        df = read_gold_mentions(datafile)
        result = evaluate(df, use_gold_mention=use_gold_mention, filename=participant, history_factor=history_factor,
                          approach=approach, mention_det=mention_det)
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
    pp_list = os.listdir(filepath)
    all_results = []
    participants = [os.path.splitext(pp)[0] for pp in pp_list]
    participants_doubled = list(zip(participants, participants))
    participants_new = list(sum(participants_doubled, ()))
    precision_recall = ['precision', 'recall']*10
    tuples = list(zip(participants_new, precision_recall))
    index = pd.MultiIndex.from_tuples(tuples)
    configs = [{'avg': 'micro', 'history': 1.0, 'approach': 'full'}, {'avg': 'macro', 'history': 1.0, 'approach': 'full'},
               {'avg': 'micro', 'history': 1.0, 'approach': 'literal'}, {'avg': 'macro', 'history': 1.0, 'approach': 'literal'},
               {'avg': 'micro', 'history': 0.0, 'approach': 'full'}, {'avg': 'macro', 'history': 0.0, 'approach': 'full'},
               {'avg': 'micro', 'history': 0.0, 'approach': 'literal'}, {'avg': 'macro', 'history': 0.0, 'approach': 'literal'}]

    for config in configs:
        results = evaluate_all_files(pp_list, filepath, use_gold_mention=False, average=config['avg'],
                                     history_factor=config['history'], approach=config['approach'], mention_det=False)
        prec_rec = list(sum(list(zip(results['precision'].values(), results['recall'].values())), ()))
        all_results.append(prec_rec)

    df = pd.DataFrame(all_results, columns=index)
    print(tabulate(df, headers='keys'))
    df.to_csv('test_nomention_04042024.csv')



