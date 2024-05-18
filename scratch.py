
from tqdm import tqdm
import json

if __name__ == '__main__':


    for num in [3]:
    
        print(f'**{num}**')

        # Format and tokenize
        with open(f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/cleaned_sample_data.json") as f:
            data = json.load(f)

        for art_id, art_dict in data.items():

            if "....and this is where stands" in art_dict["headline"].lower():

                example = art_dict
                print(art_id)

    print(json.dumps(example, indent=3))

    


