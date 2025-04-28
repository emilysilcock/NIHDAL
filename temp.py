import json

with open('/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/france_chunked.json') as f:
    new_dat = json.load(f)

with open('/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/france_chunked_old.json') as f:
    old_dat = json.load(f)

new_ids = [d['ln_id'] for d in new_dat]
old_ids = [d['ln_id'] for d in old_dat]

assert len(new_ids) == len(old_ids)

for i in new_ids:
    assert i in old_ids
