import json

def save_predictions(predictions, filename='predictions.json'):
    with open(filename, 'w') as f:
        json.dump(predictions, f, indent=4)
