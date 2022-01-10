"""funcGNN runner."""
from os.path import exists
from datetime import datetime
from utils import tab_printer
from funcgnn import funcGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a funcGNN model.
    """
    args = parameter_parser()
    trainer = funcGNNTrainer(args)
    tab_printer(args)


    if args.additional_training == 'yes' and exists("./model_state.pth"):
        trainer.fit_new_data()
        print("\nSaving Trained Model\n")
        trainer.save_model()

    if args.retrain == 'yes':
        print("\nRetraining model\n")
        trainer.fit()
        print("\nSaving Trained Model\n")
        trainer.save_model()

    elif exists("./model_state.pth"):
        print("\nPretrained model found. Loading trained model.\n")
        trainer.load_model()
        
    else:
        print("\nNo pretrained model found. Starting training...\n")
        trainer.fit()
        print("\nSaving Trained Model\n")
        trainer.save_model()

    # trainer.ROC()




if __name__ == "__main__":
    main()
