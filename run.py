from main import *
from config import *

if __name__ == '__main__':
    data_path = "./data/high_preprocess.csv"
    model_path = "./save_model/cnn_best_model.pth"
    if args.task == 'classify':
        KFold_CV(data_path)
        
    elif args.task == 'predict':
        predict_data_path = "./data/123.csv"
        predict(model_path, predict_data_path)
        
