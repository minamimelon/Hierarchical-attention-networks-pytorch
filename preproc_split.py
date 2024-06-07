import json

with open("yelp/yelp_academic_dataset_review.json", "r") as f:
    lines = f.readlines()
with open("yelp/test.json", "r") as f:
    test_set = f.readlines()
test_set_set = set(test_set)
train_set = []
cnt = 0
for x in lines:
    if x not in test_set_set:
        train_set.append(x)
    else:
        cnt = cnt +1

print(cnt, len(test_set))
with open("yelp/train.json", "w") as w:
    w.writelines(train_set[:-1000])

with open("yelp/val.json", "w") as w:
    w.writelines(train_set[-1000:])