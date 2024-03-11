import pandas as pd
import json 

x = pd.read_csv('playground/data_train/Train_Qs.csv')

data = []

for i in range(len(x)):
    query = {}

    query['id'] = f"{i}"
    if len(str(x.iloc[i]['image_id'])) == 6:
        query['image'] = f'COCO_val2014_000000{x.iloc[i]['image_id']}.jpg'
    elif len(str(x.iloc[i]['image_id'])) == 5:
        query['image'] = f'COCO_val2014_0000000{x.iloc[i]['image_id']}.jpg'
    elif len(str(x.iloc[i]['image_id'])) == 4:
        query['image'] = f'COCO_val2014_00000000{x.iloc[i]['image_id']}.jpg'
    elif len(str(x.iloc[i]['image_id'])) == 3:
        query['image'] = f'COCO_val2014_000000000{x.iloc[i]['image_id']}.jpg'
    else:
        query['image'] = f'COCO_val2014_0000000000{x.iloc[i]['image_id']}.jpg'

    conversations = [{"from": "human", "value": f"<image>\n{x.iloc[i]['question']}"}, 
                     {"from": "gpt", "value": f"{x.iloc[i]['answer']}"}]
    
    query['conversations'] = conversations

    data.append(query)


with open('playground/data_train/data.json', 'w') as f:
    json.dump(data, f, indent=2)