import json

data = json.load(open('transcripts.json', encoding='utf-8'))
failed = [x for x in data if x.get('status') == 'failed']
print(f'Failed count: {len(failed)}\n')
for x in failed:
    print(f"{x.get('video_id')}: {x.get('error')}")
