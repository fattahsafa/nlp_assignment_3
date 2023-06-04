import utils

# Define the place
place = ['London']

# Read the file and get the line count
with open('birth_dev.tsv') as file:
    lines = [line for line in file.read().split('\n') if line]

# Duplicate 'place' according to the length of lines
places = place * len(lines)

# Evaluate places
total, correct = utils.evaluate_places('birth_dev.tsv', places)

# Check if total is greater than zero to avoid division by zero
if total > 0:
    # Calculate the percentage
    percentage = (correct / total) * 100
    
    # Print the result
    print(f'Correct: {correct} out of {total}: {percentage}%')
