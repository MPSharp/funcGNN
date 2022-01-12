"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run funcGNN.")

    parser.add_argument("--embedding-size",
                        nargs="?",
                        default=200,
	                help="Embedding size of basic block.")

    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="./dataset/50k_partial_dataset/1st_half_train/",
                        # default="./dataset/smaller_partial_dataset/train/",
                        # default="./dataset/tiny_partial_dataset/train/",
	                help="Folder with training graph pair jsons.")

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="./dataset/50k_partial_dataset/1st_half_test/",
                        # default="./dataset/smaller_partial_dataset/test/",
                        # default="./dataset/tiny_partial_dataset/test/",
	                help="Folder with testing graph pair jsons.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
	                help="Number of training epochs. Default is 10.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=256,
	                help="Filters (neurons) in 1st convolution. Default is 256.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=128,
	                help="Filters (neurons) in 2nd convolution. Default is 128.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=64,
	                help="Filters (neurons) in 3rd convolution. Default is 64.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=32,
	                help="Neurons in tensor network layer. Default is 32.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=32,
	                help="Bottle neck layer neurons. Default is 32.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=512,
	                help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
	                help="Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
	                help="Dropout probability. Default is 0.2")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.add_argument("--additional_training",
                        type=str,
                        default="no",
                    help="Additional training of model. Default is 'no'.")
    parser.add_argument("--retrain",
                        type=str,
                        default="no",
                    help="Retrain the model. Default is 'no'.")

    parser.set_defaults(histogram=False)

    return parser.parse_args()
