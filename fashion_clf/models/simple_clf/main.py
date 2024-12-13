import argparse

from fashion_clf.models.simple_clf.train import train


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv_path', default='../../../data/fashion_mnist/fashion-mnist_train.csv',
                        help="Path to the training CSV file (default: %(default)s)")
    parser.add_argument('--test_csv_path', default='../../../data/fashion_mnist/fashion-mnist_test.csv',
                        help="Path to the testing CSV file (default: %(default)s)")
    parser.add_argument('--num_epoch', type=int, default=5, help="Number of epochs (default: %(default)s)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
