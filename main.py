import RNN
import LSTM
import Transformer
import pandas as pd
import argparse
import warnings
import Transformer.viz


def run_rnn(train, test, analyze, viz):
    if train:
        RNN.train.train_rnn()
    if test:
        RNN.test.test_rnn()

    if viz:
        RNN.visualize.visualize_rnn()
    if analyze:
        analysis = RNN.analyze.analyze_rnn()
        analysis.to_csv("RNN-analysis.csv")


#TODO  SLAV  match this to the interface 
def run_lstm(train,test,analyze , viz ):
    if(train):
        LSTM.train.train("data/train_data/" , "LTSM/models/")
    if(test  or viz):
        warnings.warn("Visualization and Evaluation modes are not available for LSTM model.", Warning)

    if(analyze == True): # TODO sepere test and analyze to different subroutines
        LSTM.test.analyze()



#TODO Add warnings if the paths or the models are missing for anyreason
def run_transformer(train,test,analyze , viz):
    if(train):
        Transformer.train.train_transformer()
    if(test):
        Transformer.test.evalTransformer()
    if(analyze):
        analysis = Transformer.analyze.analyze_transformer()
        analysis.to_csv("Transformer-analysis.csv")

    if(viz):
        Transformer.viz.viz_transformer()









def parse_args():
    parser = argparse.ArgumentParser(description='Run main file with specified input arguments')
    #Running modees 
    parser.add_argument('--train', action='store_true', default=False, help='allowes the user to retrain the model on the training data')
    parser.add_argument('--test', action='store_true', default=False, help='allows the user to run  the model on an evaluation data set')
    parser.add_argument('--analyze', action='store_true', default=False, help='allows the user to run analyze the model performance')
    parser.add_argument('--viz', action='store_true', default=False, help='plots the model predections vs ground truth test data ')
    # Models
    parser.add_argument('--rnn', action='store_true', default=False, help='runs the rnn model')
    parser.add_argument('--lstm', action='store_true', default=False, help='runs the lstm model')
    parser.add_argument('--t', action='store_true', default=False, help='runs the transformer model')

    return parser.parse_args()
def main():

    args = parse_args()
    train ,test, analyze ,viz , rnn , ltsm , transformer = args.train ,args.test, args.analyze , args.viz , args.rnn , args.lstm , args.t
    if(rnn):run_rnn(train , test , analyze,viz )
    if(ltsm):run_lstm(train , test, analyze,viz )
    if(transformer):run_transformer(train , test, analyze,viz )


if __name__ == '__main__':
    main()