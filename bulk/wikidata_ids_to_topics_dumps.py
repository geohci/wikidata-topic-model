import argparse
import bz2
import os
import json
from random import sample
import traceback

import fasttext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasttext_model",
                        default="../app/models/model.bin",
                        help="Location of pretrained fastText model (probably .bin file)")
    parser.add_argument("--input_qids",
                        default=None,
                        help="Input JSON file with one JSON object per row and at minimum a value under 'QID'")
    parser.add_argument("--wiki_filter",
                        nargs="*",
                        default=None,
                        help="If no input list of QIDs, only Wikidata items that have sitelinks to these languages will be listed -- e.g., 'en cs ar'.")
    parser.add_argument("--output_results",
                        default="data/example_output_data.txt",
                        help="Output JSON file with subset of input JSONs with QIDS, now with results appended.")
    parser.add_argument("--threshold",
                        default=0.5,
                        type=float,
                        help="Value at which a given topic is considered to apply for a Wikidata ID. "
                             "Defaults to 0.5 but lower values will give more topics and higher values less topics. "
                             "Set threshold to 0 for full model output.")
    args = parser.parse_args()

    # fastText model for providing predicted labels to Wikidata items
    try:
        model = fasttext.load_model(args.fasttext_model)
    except ValueError:
        print("Could not load model at location: {0}\n".format(os.path.abspath(args.fasttext_model)))
        print("Check to make sure that file exists and is not just a git LFS pointer.")
        traceback.print_exc()
        return

    # if input JSON provided, only these Wikidata items will be processed
    wd_items_to_query = None
    if args.input_qids:
        print("Loading QIDs to analyze from {0}".format(args.input_qids))
        wd_items_to_query = set()
        with open(args.input_qids, 'r') as fin:
            for i, line in enumerate(fin, start=1):
                try:
                    wd_item = json.loads(line.strip())
                except json.decoder.JSONDecodeError:
                    print("Invalid line ({0}): {1}".format(i, line.strip()))
                    continue
                if 'QID' in wd_item:
                    wd_items_to_query.add(wd_item['QID'])

    items_processed = 0
    with bz2.open(args.output_results, 'wt') as fout:
        if args.threshold > 0:
            print("Only providing labels with probability >= {0}".format(args.threshold))
        for qid, titles, claims_str, disamb_list, has_coords, human in loop_through_wd_dump(
                qids=wd_items_to_query, sites=args.wiki_filter):
            items_processed += 1
            # make prediction
            lbls, scores = model.predict(claims_str, k=-1)
            results = {l:s for l,s in zip(lbls, scores)}
            # adjust model output according to a few rules to better match intuitions
            if disamb_list:
                # identify disambiguation pages and lists explicitly
                results['Compilation.List_Disambig'] = 1
            if human:
                # language/literature should not include people (just actual biographies)
                l = '__label__Culture.Language_and_literature'
                results[l] = max(0, results[l] - 0.501)
                results['Person'] = 1
            if not has_coords:
                # geography should only be applied to topics w/ actual physical locations
                geo_keys = [k for k in results if k.startswith('__label__Geography')]
                for l in geo_keys:
                    results[l] = max(0, results[l] - 0.501)

            # build high-level category results (e.g., STEM, Geography, Culture)
            # this depends on the assumption that predicted labels are independent, which is clearly wrong
            # for instance, the model likely has correlated errors when it comes to things like STEM.Technology and STEM.Engineering
            # this is the best I can do though currently without building a separate high-level topics model
            # in practice, the high-level results tend to make sense
            hlc_results = {}
            for l in results:
                hlc = ft_to_toplevel(l)
                hlc_results[hlc] = hlc_results.get(hlc, 1) * (1 - results[l])
            hlc_results = {l:1-p for l,p in hlc_results.items()}

            # sort and filter results to just those above threshold
            sorted_res = [(l.replace("__label__", ""), round(results[l], 4)) for l in sorted(results, key=results.get, reverse=True)]
            sorted_hlc_res = [(l, round(hlc_results[l], 4)) for l in sorted(hlc_results, key=hlc_results.get, reverse=True)]
            if args.threshold > 0:
                sorted_res = [(l, p) for l,p in sorted_res if p > args.threshold]
                sorted_hlc_res = [(l, p) for l,p in sorted_hlc_res if p > args.threshold]
            output_json = {'qid':qid, 'titles':titles, 'predicted_mid_labels':sorted_res, 'predicted_top_labels':sorted_hlc_res}

            fout.write(json.dumps(output_json) + '\n')
            if items_processed % 100000 == 0:
                print("{0} items processed. Last item: {1}".format(items_processed, output_json))

def ft_to_toplevel(fasttext_lbl):
    """Example: '__label__STEM.Technology' -> 'STEM'"""
    return fasttext_lbl.replace('__label__','').split('.')[0]

def tuple_to_ft_format(claims_tuples):
    """Example: [(P31:Q5), (P625,), ...] -> 'P31 Q5 P625 ...'"""
    return ' '.join([' '.join(c) for c in sample(claims_tuples, len(claims_tuples))])

def loop_through_wd_dump(qids=None, sites=None):
    """Get Wikidata claims for items that match filters."""
    items_written = 0
    indexerror = 0
    disamb_list_vals = ['Q4167410', 'Q13406463']
    dump_fn = '/mnt/data/xmldatadumps/public/wikidatawiki/entities/latest-all.json.bz2'
    print("Making topic predictions based on {0}".format(dump_fn))
    if qids is not None:
        print("Filtering down to {0} QIDs provided.".format(len(qids)))
    elif sites is not None:
        sites = set(sites)
        print("Site filter: {0}".format(sites))
    else:
        print("Processing all Wikidata items with any wiki sitelinks.")
    with bz2.open(dump_fn, 'rt') as fin:
        for idx, line in enumerate(fin, start=1):
            try:
                item_json = json.loads(line[:-2])
            except Exception:
                try:
                    item_json = json.loads(line)
                except Exception:
                    print("Error:", idx, line)
                    continue
            if idx % 100000 == 0:
                print("{0} lines processed. {1} kept. {2} index errors".format(idx, items_written, indexerror))

            # filtering
            qid = item_json.get('id', None)
            if qids is not None and qid not in qids:
                continue
            titles = {l[:-4]:item_json['sitelinks'][l].get('title', None) for l in item_json.get('sitelinks', []) if
                      l.endswith('wiki') and l != 'commonswiki' and l != 'specieswiki'}
            if not titles:
                continue
            if sites is not None and not sites.intersection(titles):
                continue

            disamb_list = False
            has_coords = False
            human = False
            claims = item_json.get('claims', {})
            claim_tuples = []
            for property in claims:  # each property, such as P31 instance-of
                included = False
                for statement in claims[property]:  # each value under that property -- e.g., instance-of might have three different values
                    try:
                        if statement['type'] == 'statement' and statement['mainsnak']['datatype'] == 'wikibase-item':
                            val = statement['mainsnak']['datavalue']['value']['id']
                            claim_tuples.append((property, val))
                            included = True
                            if property == 'P31':
                                if val in disamb_list_vals:
                                    disamb_list = True
                                elif val == 'Q5':
                                    human = True
                    except Exception:
                        indexerror += 1
                if not included:
                    claim_tuples.append((property,))
                    if property == 'P625':
                        has_coords = True
                    elif property == 'P360':
                        disamb_list = True
            if not len(claim_tuples):
                claim_tuples = [('<NOCLAIM>',)]
            yield qid, titles, tuple_to_ft_format(claim_tuples), disamb_list, has_coords, human

if __name__ == "__main__":
    main()