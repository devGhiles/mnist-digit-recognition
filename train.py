import argparse
import json

import pandas as pd

from core import CNNKerasEncoder, CNNKerasModel,\
                 FeedForwardNeuralNetworkModel, FeedForwardNeuralNetworkEncoder


model_type_to_model_class = {
    'cnn': CNNKerasModel,
    'ff': FeedForwardNeuralNetworkModel
}


model_type_to_encoder_class = {
    'cnn': CNNKerasEncoder,
    'ff': FeedForwardNeuralNetworkEncoder
}


def main(model_type, train_filepath, valid_filepath, model_filepath,
    training_report_filepath, model_hyperparameters, training_parameters):
    # Get the classes corresponding to the model type
    try:
        model_class = model_type_to_model_class[model_type]
        encoder_class = model_type_to_encoder_class[model_type]
    except KeyError:
        print('Unknown model type: {}'.format(model_type))
        return

    # Read the data
    print('Read training and validation data...')
    train_df = pd.read_csv(train_filepath, sep=',', encoding='utf-8')
    valid_df = pd.read_csv(valid_filepath, sep=',', encoding='utf-8')

    # Encode the data
    print('Encode the training and validation data...')
    encoder = encoder_class()
    X_train, y_train = encoder.encode_X_and_y(train_df)
    X_valid, y_valid = encoder.encode_X_and_y(valid_df)

    # Launch the training
    print('Build and train the model...')
    model = model_class(**model_hyperparameters)
    training_report = model.train(X_train,
                                  y_train,
                                  X_valid=X_valid,
                                  y_valid=y_valid,
                                  **training_parameters)

    # Save the trained model on disk
    print('Save the trained model...')
    model.save(model_filepath)

    # Save the training report on disk
    print('Save the training report...')
    training_report.save(training_report_filepath)


if __name__ == '__main__':
    # Create the cli arguments parser
    parser = argparse.ArgumentParser('Train a model and save it on disk along with its training report.')
    parser.add_argument('model_type', choices=model_type_to_model_class.keys(), help='The type of the model.')
    parser.add_argument('train_file', help='File path of the training file.')
    parser.add_argument('valid_file', help='File path of the validation file.')
    parser.add_argument('model_file', help='File path where to save the trained model.')
    parser.add_argument('training_report_file', help='File path where to save the training report.')
    parser.add_argument('parameters_file', help='File path to the json file describing the model and training parameters.')

    # Parse the cli arguments
    args = parser.parse_args()
    model_type = args.model_type
    train_filepath = args.train_file
    valid_filepath = args.valid_file
    model_filepath = args.model_file
    training_report_filepath = args.training_report_file
    parameters_filepath = args.parameters_file

    # Read the parameters file
    try:
        with open(parameters_filepath, 'r', encoding='utf-8') as f:
            parameters = json.load(f)

        model_hyperparameters = parameters['model_hyperparameters']
        training_parameters = parameters['training_parameters']

        main(model_type, train_filepath, valid_filepath, model_filepath,
            training_report_filepath, model_hyperparameters, training_parameters)

    except FileNotFoundError:
        print('Parameters file {} not found.'.format(parameters_filepath))
