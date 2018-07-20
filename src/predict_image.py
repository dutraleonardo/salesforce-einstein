import json
import time
from vision.prediction_creator import Prediction
from constants import ACCESS_TOKEN


def main():

    url_list = []
    access_token = ACCESS_TOKEN
    with open('static/food.csv', 'rU') as csvfile:
        reader = (line.replace('\r', '') for line in csvfile)
        for row in reader:
            url_list.append(row)
    model_id = 'FoodImageClassifier'
    prediction = Prediction(access_token=access_token)

    try:
        for url in url_list[1:]:
            response = prediction.predict_remote_image(url, model_id)
            probabilities = response['probabilities']
            print(url)
            print('#' * 50)
            for x in probabilities:
                print(str(x['label']) + ": " + str(x['probability']))
            print('#'*50)

            time.sleep(3)

    except TypeError:
        print('response ok? ' + str(response.ok))
        print('response content: ' + str(response.content))
    return True


if __name__ == "__main__":
    main()