import json

with open("yelp/yelp_academic_dataset_review.json", "r") as f:
    lines = f.readlines()
with open("yelp/test.json", "r") as f:
    test_set = f.readlines()
test_set_set = set(test_set)
train_set = []
cnt = 0
with open("yelp/train.json", "w") as w:
    for x in lines:
        if x not in test_set_set:
            train_set.append(x)
        else:
            cnt = cnt +1
    w.writelines(train_set)
    print(cnt, len(test_set))

with open("data/test.csv", "w") as f:
    for x in test_set:
        js = json.loads(x)
        f.write('"{}","{}"\n'.format(js["stars"],str(js["text"]).replace("\n","\\n")))

with open("data/train.csv", "w") as f:
    for x in train_set:
        js = json.loads(x)
        f.write('"{}","{}"\n'.format(js["stars"],str(js["text"]).replace("\n","\\n")))