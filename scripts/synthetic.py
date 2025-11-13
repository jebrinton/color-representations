import json

with open("data/synthetic.json", "r") as f:
    data = json.load(f)

print(data)

for template in data["templates"]:
    for object in data["objects"]:
        for color in data["colors"]:
            for number in data["numbers"]:
                print(template.format(object=object, color=color, number=number))