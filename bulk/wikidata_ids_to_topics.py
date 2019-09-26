import argparse
import os
import json
from random import sample
import traceback

import fasttext
import mwapi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasttext_model",
                        default="../app/models/model.bin",
                        help="Location of pretrained fastText model (probably .bin file)")
    parser.add_argument("--input_qids",
                        default="data/example_input_data.txt",
                        help="Input JSON file with one JSON object per row and at minimum a value under 'QID'")
    parser.add_argument("--output_results",
                        default="data/example_output_data.txt",
                        help="Output JSON file with subset of input JSONs with QIDS (in order) with results appended.")
    parser.add_argument("--threshold",
                        default=0.5,
                        type=float,
                        help="Value at which a given topic is considered to apply for a Wikidata ID. "
                             "Defaults to 0.5 but lower values will give more topics and higher values less topics. "
                             "Set threshold to 0 for full model output.")
    parser.add_argument("--query_limit",
                        default=50,
                        type=int,
                        help="Number of Wikidata IDs to process at once -- i.e. per API call. Max 50.")
    args = parser.parse_args()

    try:
        model = fasttext.load_model(args.fasttext_model)
    except ValueError:
        print("Could not load model at location: {0}\n".format(os.path.abspath(args.fasttext_model)))
        traceback.print_exc()
        return

    session = mwapi.Session('https://www.wikidata.org',
                            user_agent='wikidata topic app -- isaac@wikimedia.org')

    items_processed = 0
    items_skipped = 0
    with open(args.input_qids, 'r') as fin:
        with open(args.output_results, 'w') as fout:
            wd_items_to_query = []
            for i, line in enumerate(fin, start=1):
                try:
                    wd_item = json.loads(line.strip())
                except json.decoder.JSONDecodeError:
                    print("Invalid line ({0}): {1}".format(i, line.strip()))
                    items_skipped += 1
                    continue
                if 'QID' in wd_item:
                    items_processed += 1
                    wd_items_to_query.append(wd_item)
                    # process 50 items at a time to reduce API load
                    if len(wd_items_to_query) == args.query_limit:
                        print("Processing items {0} through {1} ({2} skipped so far)".format(items_processed - args.query_limit,
                                                                                             items_processed, items_skipped))
                        label_qids(wd_items_to_query, session, model, args.threshold)
                        for qid_json in wd_items_to_query:
                            fout.write(json.dumps(qid_json) + "\n")
                        wd_items_to_query = []
                else:
                    items_skipped += 1
            if wd_items_to_query:
                print("Processing final items {0} through {1} ({2} skipped so far)".format(
                    items_processed - len(wd_items_to_query), items_processed, items_skipped))
                label_qids(wd_items_to_query, session, model, args.threshold)
                for qid_json in wd_items_to_query:
                    fout.write(json.dumps(qid_json) + "\n")


def label_qids(wd_items_to_query, session, model, threshold=0.5):
    # build QID list
    qids = [item['QID'] for item in wd_items_to_query]
    qid_to_idx = {qid:idx for idx, qid in enumerate(qids)}
    qids_str = "|".join(qids)

    # get claims for wikidata item
    try:
        result = session.get(
            action="wbgetentities",
            props='claims',
            format='json',
            ids=qids_str
        )
    except Exception:
        print("Failed:", qids_str)
        return

    for entity in result['entities']:
        qid = result['entities'][entity]['id']
        # convert claims to fastText bag-of-words format
        claims = result['entities'][entity]['claims']
        claims_tuples = []
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
        claims_str = ' '.join([' '.join(c) for c in sample(claims_tuples, len(claims_tuples))])

        # make prediction
        lbls, scores = model.predict(claims_str, k=-1)
        results = {l:s for l,s in zip(lbls, scores)}
        sorted_res = [(l.replace("__label__", ""), results[l]) for l in sorted(results, key=results.get, reverse=True)]
        above_threshold = [r for r in sorted_res if r[1] >= threshold]

        # add results to input list of wikidata items
        wd_items_to_query[qid_to_idx[qid]]['labels'] = above_threshold


if __name__ == "__main__":
    main()