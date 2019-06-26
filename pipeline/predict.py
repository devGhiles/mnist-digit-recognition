import argparse
import sys

import pandas as pd

from pipeline.core import CNNKerasModel, CNNKerasEncoder


model_type_to_model_class = {
    'cnn': CNNKerasModel
}


def main(model_type, model_filepath, test_filepath, submission_filepath):
    # get the model class
    try:
        model_class = model_type_to_model_class[model_type]
    except KeyError:
        print('Unrecognized model type: {}'.format(model_type), file=sys.stderr)

    print('Load the model...')
    model = model_class()
    model.load(model_filepath)

    print('Read the test data...')
    test_df = pd.read_csv(test_filepath, sep=',', encoding='utf-8')

    print('Encode the test data...')
    encoder = CNNKerasEncoder()
    X_test = encoder.encode_X(test_df)

    print('Make the predictions...')
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)

    print('Create the submission file...')
    submission_df = {'ImageId': range(1, len(y_pred) + 1), 'Label': y_pred}
    submission_df = pd.DataFrame(submission_df)[['ImageId', 'Label']]
    submission_df.to_csv(submission_filepath, sep=',', encoding='utf-8', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate a submission file for a test dataset from a saved model.')
    parser.add_argument('type', choices=['cnn'], help='Model type.')
    parser.add_argument('model_file', help='File path to the saved model.')
    parser.add_argument('test_file', help='File path to the test file.')
    parser.add_argument('submission_file', help='File path where to save the submission file.')

    args = parser.parse_args()
    model_type = args.type
    model_filepath = args.model_file
    test_filepath = args.test_file
    submission_filepath = args.submission_file

    main(model_type, model_filepath, test_filepath, submission_filepath)
