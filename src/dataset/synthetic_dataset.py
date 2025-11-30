import json
from torch.utils.data import Dataset
import random

class SyntheticDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r") as f:
            self.data = json.load(f)
        

    def get_contastive_pairs(self, template_type, color_type):
        contrastive_data = {}
        for color in self.data[color_type]:
            contrastive_data[color] = {"positive": [], "negative": []}
            for template in self.data[template_type]:
                for object_ in self.data["objects"]:
                    for number in self.data["numbers"]:
                        sent = template.format(object=object_, color=color, number=number)
                        contrastive_data[color]['positive'].append(sent)
        
        for pos_color in contrastive_data.keys():

            neg_colors = list(set(list(contrastive_data.keys())) - set([pos_color]))
            
            all_neg_sentences = []
            for neg_color in neg_colors:
                all_neg_sentences.extend(contrastive_data[neg_color]['positive'])
            
            sampled_sentences = random.sample(all_neg_sentences, len(contrastive_data[pos_color]['positive'])) 
            
            contrastive_data[pos_color]['negative'] = sampled_sentences
            

        return contrastive_data
    
    def get_color_sentences(self, template_type, color_type):
        output = {}
        for color in self.data[color_type]:
            output[color] = []
            for template in self.data[template_type]:
                for object_ in self.data["objects"]:
                    for number in self.data["numbers"]:
                        sent = template.format(object=object_, color=color, number=number)
                        output[color].append(sent)
            # output[color] = output[color]
        
        return output

        
class CommonObjects(Dataset):
    def __init__(self, path: str):
        with open(path, "r") as f:
            self.data = json.load(f)
    
    def get_all_common_objects(self):
        # print(self.data["objects"])
        data = [
            f"Answer in one word. What is the color of {i}?"
            for i in self.data.keys()
        ]

        return data


# x = CommonObjects("/projectnb/cs599m1/projects/color-rep/color-representations/data/common_objects.json")
# print(x.get_all_common_objects()[:1])