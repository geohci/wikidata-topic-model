# wikidata-topic-model
Map Wikidata items to a taxonomy of topics from WikiProjects. This approach represents a Wikipedia article based on the claims contained in its Wikidata item. The topics are determined based on the WikiProject directory. Currently this repository just contains a Flask app that provides predictions based on a pre-trained model but eventually it will be expanded to include the entire training process.

## Running the app

```
cd app
python3 app.py
```
NOTE: you must have the fastText Python module installed.
See https://fasttext.cc/docs/en/support.html for how to install.

### Querying Wikidata items
Queries can be made via the browser. For example, for [Toni Morrison](https://www.wikidata.org/wiki/Q72334):

http://127.0.0.1:5000/api/v1/wikidata/topic?qid=Q72334

The threshold above which a topic is returned [0-1] can be set via the `threshold` parameter but otherwise defaults to `0.5`:

http://127.0.0.1:5000/api/v1/wikidata/topic?qid=Q72334&threshold=0.1

Append the `debug` parameter for additional output including all of the topics and scores and the Wikidata claims processed by the model:

http://127.0.0.1:5000/api/v1/wikidata/topic?qid=Q72334&debug

## Running the bulk Wikidata script
This script takes in a file with JSON objects containing the wikidata IDs to query (and any additional metadata) and outputs these JSONs with the predicted labels. Example input / output data is provided in the `bulk/data` folder.

```
cd bulk
python3 wikidata_ids_to_topics.py --help
python3 wikidata_ids_to_topics.py
```
NOTE: like the app, you must have the fastText Python module installed.
See https://fasttext.cc/docs/en/support.html for how to install.

## See Also
https://meta.wikimedia.org/wiki/Research_talk:Characterizing_Wikipedia_Reader_Behaviour/Demographics_and_Wikipedia_use_cases/Work_log/2019-09-11
