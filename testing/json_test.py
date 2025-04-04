import json


## Loading JSON

json_string = '{"name": "Matt", "age": 23, "isStudent": false}'
data = json.loads(json_string)

print(data)
print(data["name"])  # Output: Alice


## Writing JSON

person = {
    "name": "Bob",
    "age": 30,
    "isStudent": True
}

json_output = json.dumps(person)
print(json_output)