import re

import fasttext
from flask import Flask, request, jsonify, render_template
import mwapi

app = Flask(__name__)
app.config["DEBUG"] = True
SESSION = mwapi.Session('https://www.wikidata.org',
                        user_agent='wikidata topic app -- isaac@wikimedia.org')
FT_MODEL = fasttext.load_model('models/model.bin')

PROVIDE_EXPLANATIONS = False

print("Try: http://127.0.0.1:5000/api/v1/wikidata/topic?qid=Q72334&debug")

if PROVIDE_EXPLANATIONS:
    from lime.lime_text import LimeTextExplainer
    import numpy as np

    lbls_to_idx = {l.replace('__label__', ''):i for i,l in enumerate(FT_MODEL.get_labels())}
    EXPLAINER = LimeTextExplainer(class_names=[l.replace('__label__', '') for l in FT_MODEL.get_labels()])

@app.route('/')
def index():
    return 'Server Works!'


def adjust_topics_based_on_claims(topics, claims):
    joined_claims = [":".join(c) for c in claims]
    properties = [c[0] for c in claims]
    # list / disambiguation pages
    if 'P31:Q4167410' in joined_claims:
        topics = [('Compilation.List_Disambig', 1, 'P31:Q4167410 -- Disambiguation')] + topics
    elif 'P31:Q13406463' in joined_claims:
        topics = [('Compilation.List_Disambig', 1, 'P31:Q13406463 -- List')] + topics
    elif 'P360' in properties:
        topics = [('Compilation.List_Disambig', 1, 'P360 -- List')] + topics
    # geography only should apply to items with coordinates
    if 'P625' not in joined_claims:
        for idx in range(len(topics)):
            if topics[idx][0].startswith('Geography'):
                topics[idx] = (topics[idx][0], max(0, topics[idx][1] - 0.501), ' -- '.join([topics[idx][2], "downgraded bc no coords"]))
    # language/literature should not include people (just actual biographies)
    if 'P31:Q5' in joined_claims:
        for idx in range(len(topics)):
            if topics[idx][0] == 'Culture.Language_and_literature':
                topics[idx] = (topics[idx][0], max(0, topics[idx][1] - 0.501), ' -- '.join([topics[idx][2], " downgraded bc human"]))
        topics = [('Person', 1, 'P31:Q5 (Human)')] + topics
    topics = sorted(topics, key=lambda tup: tup[1], reverse=True)
    return topics, claims


@app.route('/api/v1/wikidata/topic', methods=['GET'])
def get_topics():
    qid, threshold, debug = validate_api_args()
    name, topics, claims = label_qid(qid, SESSION, FT_MODEL, threshold)
    topics, claims = adjust_topics_based_on_claims(topics, claims)
    if debug:
        return render_template('wikidata_topics.html',
                               qid=qid, claims=claims, topics=topics, name=name)
    else:
        topics = [{'topic':t[0], 'score':t[1], 'explanation':t[2]} for t in topics]
        return jsonify(topics)

def validate_api_args():
    if 'qid' in request.args:
        qid = request.args['qid'].upper()
    else:
        return "Error: no 'qid' field provided. Please specify."

    if not re.match('^Q[0-9]+$', qid):
        return "Error: poorly formatted 'qid' field. {0} does not match 'Q#...'".format(qid)

    threshold = 0.5
    if 'threshold' in request.args:
        try:
            threshold = float(request.args['threshold'])
        except ValueError:
            return "Error: threshold value provided not a float: {0}".format(request.args['threshold'])

    debug = False
    if 'debug' in request.args:
        debug = True
        threshold = 0

    return qid, threshold, debug


def predict_proba_lime(datapoints):
    lbl_to_idx = {l:i for i,l in enumerate(FT_MODEL.get_labels())}
    probabilities = np.zeros((len(datapoints), len(lbl_to_idx)))
    for i, dp in enumerate(datapoints):
        lbls, probs = FT_MODEL.predict(dp, k=-1)
        for lbl_idx, lbl in enumerate(lbls):
            probabilities[i][lbl_to_idx[lbl]] = probs[lbl_idx]
    return probabilities


def label_qid(qid, session, model, threshold=0.5, debug=False):
    # default results
    name = ""
    above_threshold = []
    claims_tuples = []

    # get claims for wikidata item
    result = {}
    try:
        result = session.get(
            action="wbgetentities",
            props='claims|labels',
            languages='en',
            languagefallback='',
            format='json',
            ids=qid
        )
    except Exception:
        print("Failed:", qid)
    if debug:
        print(result)

    if 'missing' in result['entities'][qid]:
        print("No results:", qid)
    else:
        # get best label
        for lbl in result['entities'][qid]['labels']:
            name = result['entities'][qid]['labels'][lbl]['value']
            print('{0}: {1}'.format(qid, name))
            break

        # convert claims to fastText bag-of-words format
        claims = result['entities'][qid]['claims']
        for prop in claims:  # each property, such as P31 instance-of
            included = False
            for statement in claims[prop]:  # each value under that property -- e.g., instance-of might have three different values
                try:
                    if statement['type'] == 'statement' and statement['mainsnak']['datatype'] == 'wikibase-item':
                        claims_tuples.append((prop, statement['mainsnak']['datavalue']['value']['id']))
                        included = True
                except Exception:
                    continue
            if not included:
                claims_tuples.append((prop, ))
        if not len(claims_tuples):
            claims_tuples = [('<NOCLAIM>', )]
        if debug:
            print(claims_tuples)
        claims_str = ' '.join([' '.join(c) for c in claims_tuples])

        # make prediction
        lbls, scores = model.predict(claims_str, k=-1)
        results = {l:s for l,s in zip(lbls, scores)}
        if debug:
            print(results)
        sorted_res = [(l.replace("__label__", ""), results[l], "None") for l in sorted(results, key=results.get, reverse=True)]
        above_threshold = [r for r in sorted_res if r[1] >= threshold]
        lbls_above_threshold = []
        if above_threshold:
            for res in above_threshold:
                print('{0}: {1:.3f} -- {2}'.format(*res))
                if res[1] > 0.5:
                    lbls_above_threshold.append(res[0])
        else:
            print("No label above {0} threshold.".format(threshold))
            print("Top result: {0} ({1:.3f}) -- {2}".format(sorted_res[0][0], sorted_res[0][1], sorted_res[0][2]))

        if PROVIDE_EXPLANATIONS and lbls_above_threshold:
            exp = EXPLAINER.explain_instance(claims_str, predict_proba_lime, num_features=5,
                                             labels=[lbls_to_idx[l] for l in lbls_above_threshold])
            for i,lbl in enumerate(lbls_above_threshold):
                above_threshold[i] = (above_threshold[i][0], above_threshold[i][1],
                                      '; '.join(['{0} ({1:.3f})'.format(ft[0], ft[1]) for ft in exp.as_list(label=lbls_to_idx[lbl])]))
                print(above_threshold[i])

    return name, above_threshold, claims_tuples


app.run()