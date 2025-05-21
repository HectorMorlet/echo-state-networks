import json
from collections import defaultdict

# Read the tests file
with open('tests.json', 'r') as f:
    tests = json.load(f)

def normalize_value(value):
    """Normalize a value by sorting any lists/arrays it contains"""
    if isinstance(value, list):
        return sorted(normalize_value(x) for x in value)
    elif isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    return value

# Helper function to get test key (all params except error_funcs, errors, and date)
def get_test_key(test):
    key_dict = test.copy()
    key_dict.pop('error_funcs', None)
    key_dict.pop('errors', None)
    key_dict.pop('date', None)
    # Normalize all values to handle arrays in different orders
    key_dict = normalize_value(key_dict)
    return json.dumps(key_dict, sort_keys=True)

# Helper function to compare error functions ignoring order
def compare_error_funcs(funcs1, funcs2):
    return set(funcs1) == set(funcs2)

# Helper function to compare errors ignoring order in arrays
def compare_errors(errors1, errors2):
    if set(errors1.keys()) != set(errors2.keys()):
        return False
    for key in errors1:
        if not isinstance(errors1[key], list) or not isinstance(errors2[key], list):
            return False
        if len(errors1[key]) != len(errors2[key]):
            return False
        if sorted(errors1[key]) != sorted(errors2[key]):
            return False
    return True

# Group tests by their key
grouped_tests = defaultdict(list)
for test in tests:
    key = get_test_key(test)
    grouped_tests[key].append(test)

# Check for duplicates
found_duplicates = False
for key, test_group in grouped_tests.items():
    if len(test_group) > 1:
        # Check if the tests are actually duplicates considering array order
        for i in range(len(test_group)):
            for j in range(i + 1, len(test_group)):
                test1, test2 = test_group[i], test_group[j]
                if (compare_error_funcs(test1['error_funcs'], test2['error_funcs']) and
                    compare_errors(test1['errors'], test2['errors'])):
                    if not found_duplicates:
                        print("\nFound duplicates (considering arrays with same elements as equal):")
                    found_duplicates = True
                    print("\nDuplicate pair:")
                    print(f"Test 1:")
                    print(f"- Date: {test1.get('date', 'No date')}")
                    print(f"- Error functions: {test1['error_funcs']}")
                    print(f"- n_steps: {test1.get('n_steps', [])}")
                    print(f"- Errors: {test1['errors']}")
                    print(f"\nTest 2:")
                    print(f"- Date: {test2.get('date', 'No date')}")
                    print(f"- Error functions: {test2['error_funcs']}")
                    print(f"- n_steps: {test2.get('n_steps', [])}")
                    print(f"- Errors: {test2['errors']}")
                    print("\nShared parameters:")
                    shared_params = json.loads(key)
                    print(json.dumps(shared_params, indent=2))
                    print("-" * 80)

if not found_duplicates:
    print("No duplicates found in tests.json (even when ignoring array order)") 