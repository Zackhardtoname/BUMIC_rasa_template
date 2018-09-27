from fuzzywuzzy import fuzz
from fuzzywuzzy import process

target = "deadline"
l = ['study', 'homework', 'gender', 'machine']
r = []
for com in l:
    r.append(fuzz.ratio(target, com))
print(r)