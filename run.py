import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--job-name", type=str, default="testing")
parser.add_argument("--nprocs", type=int, default=20)
parser.add_argument("--partition", type=str, default="nodo.q")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--dataset", type=str, default="mnist")
args = parser.parse_args()

print(args.epochs, args.dataset)
