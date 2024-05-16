
from tqdm import tqdm
import json

if __name__ == '__main__':


    for num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    
        print(f'**{num}**')

        # Format and tokenize
        with open(f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/cleaned_sample_data.json") as f:
            data = json.load(f)

        for art_id, art_dict in tqdm(data.items()):

            if "oap a rapist" in art_dict["headline"].lower():

                print(f'********************{num}*************************')
