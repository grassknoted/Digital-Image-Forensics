import csv

image_classes = {}

with open('sample_submission.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        image_classes[str(row[0])] = []

with open('fft1_predictions.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        image_classes[str(row[0])].append(str(row[1]))

with open('fft2_predictions.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        image_classes[str(row[0])].append(str(row[1]))

with open('fft3_predictions.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        image_classes[str(row[0])].append(str(row[1]))

with open('fft4_predictions.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        image_classes[str(row[0])].append(str(row[1]))

with open('all_fft_predictions.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        image_classes[str(row[0])].append(str(row[1]))

def leaders(xs, top=10):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]

for dict_keys in image_classes:
    for predictions in image_classes[dict_keys]:
        predictions = leaders(predictions)
