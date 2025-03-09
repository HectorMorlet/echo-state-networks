import json
import os

# Read the files
with open('tests - old.json', 'r') as f:
    old_tests = json.load(f)
with open('tests.json', 'r') as f:
    current_tests = json.load(f)

# Helper function to get test key (all params except error_func/error_funcs, errors, and date)
def get_test_key(test):
    key_dict = test.copy()
    key_dict.pop('error_func', None)
    key_dict.pop('error_funcs', None)
    key_dict.pop('errors', None)
    key_dict.pop('date', None)  # Ignore date when comparing tests
    return json.dumps(key_dict, sort_keys=True)

# Helper function to combine tests with same parameters
def combine_tests(tests):
    if len(tests) == 0:
        return []
    
    grouped_tests = {}
    for test in tests:
        key = get_test_key(test)
        if key not in grouped_tests:
            grouped_tests[key] = []
        grouped_tests[key].append(test)
    
    combined = []
    for tests_group in grouped_tests.values():
        base_test = tests_group[0].copy()
        error_funcs = []
        errors = {}
        latest_date = base_test.get('date', '')
        
        for test in tests_group:
            if 'error_func' in test:
                error_funcs.append(test['error_func'])
                errors[test['error_func']] = test['errors']
            elif 'error_funcs' in test:
                error_funcs.extend(test['error_funcs'])
                errors.update(test['errors'])
            
            # Keep the latest date
            test_date = test.get('date', '')
            if test_date > latest_date:
                latest_date = test_date
        
        if 'error_func' in base_test:
            base_test.pop('error_func')
        base_test['error_funcs'] = list(set(error_funcs))  # Remove duplicates
        base_test['errors'] = errors
        if latest_date:
            base_test['date'] = latest_date
        combined.append(base_test)
    
    return combined

# First combine any tests in current_tests that should be combined
current_tests = combine_tests(current_tests)

# Then process and combine old tests
old_format_tests = [test for test in old_tests if 'error_func' in test]
new_tests = combine_tests(old_format_tests)

# Combine all tests and remove any duplicates
all_tests = current_tests + new_tests
combined_tests = combine_tests(all_tests)

# Write back to tests.json
with open('tests.json', 'w') as f:
    json.dump(combined_tests, f) 