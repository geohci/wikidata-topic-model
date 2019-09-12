import re

import fasttext
from flask import Flask, request, jsonify, render_template
import mwapi

app = Flask(__name__)
app.config["DEBUG"] = True
SESSION = mwapi.Session('https://www.wikidata.org',
                        user_agent='wikidata topic app -- isaac@wikimedia.org')
FT_MODEL = fasttext.load_model('models/model.bin')

@app.route('/')
def index():
    return 'Server Works!'


@app.route('/api/v1/wikidata/topic', methods=['GET'])
def get_topics():
    qid, threshold, debug = validate_api_args()
    name, topics, claims = label_qid(qid, SESSION, FT_MODEL, threshold)
    if debug:
        return render_template('wikidata_topics.html',
                               qid=qid, claims=claims, topics=topics, name=name)
    else:
        topics = [{'topic':t[0], 'score':t[1]} for t in topics]
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
        sorted_res = [(l.replace("__label__", ""), results[l]) for l in sorted(results, key=results.get, reverse=True)]
        above_threshold = [r for r in sorted_res if r[1] >= threshold]
        if above_threshold:
            for res in above_threshold:
                print('{0}: {1}'.format(*res))
        else:
            print("No label above {0} threshold.".format(threshold))
            print("Top result: {0} ({1:.4f})".format(sorted_res[0][0], sorted_res[0][1]))

    return name, above_threshold, claims_tuples


app.run()