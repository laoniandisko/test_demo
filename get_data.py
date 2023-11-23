import json
from tqdm import tqdm
from predict import getMask

def get_data(json_path,model_path):
    with open(json_path, 'rb') as f:
        json_data = json.load(f)
    tbar = tqdm(json_data)
    for idx, item in enumerate(tbar):
        img_path = item['img_name']
        mask = f"{item['segment_id']}.png"
        sents = [i['sent'] for i in item['sentences']]
        print(img_path,mask,sents)
        exit()

if __name__ == "__main__":
    get_data('dataset/anns/refcoco/testA.json','.')
