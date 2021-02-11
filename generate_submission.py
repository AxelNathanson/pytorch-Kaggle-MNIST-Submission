import csv
from datetime import datetime


def generate_submission(test_classification):
    time = str(datetime.now())
    time = time[:10] + '.' + time[11:13] + '.' + time[14:16]
    file_name = f'submissions/submission{time}.csv'

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageId', 'Label'])

        for case, prediction in enumerate(test_classification, 1):
            writer.writerow([case, prediction])






