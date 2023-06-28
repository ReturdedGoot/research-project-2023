import pandas as pd
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="", help="Directory path of the folder to analyze")
    parser.add_argument("--file", type=str, default="", help="Name of the file to analyze")
    parser.add_argument("--acc", action="store_true", help="include test accuracy in the plot")
    parser.add_argument("--loss", action="store_true", help="include loss in the plot")
    parser.add_argument("--xlab", default="steps", help="X label for the plot")
    parser.add_argument("--ylab", default="", help="Y label for the plot")
    parser.add_argument("--title", default="", help="title for the plot")
    parser.add_argument("--show", action="store_true", help="Displays the plot")
    parser.add_argument("--savefile", default="", help="The filename to save the plot as")
    args = parser.parse_args()
    if args.directory != "":
        filepath = Path(args.directory + "/" + args.file + ".csv") if "csv" not in args.file else Path(
            args.directory + "/" + args.file)
    else:
        filepath = Path(args.file + ".csv") if "csv" not in args.file else Path(args.file)

    plt.figure()
    plt.title(args.title)
    df = pd.read_csv(filepath)
    x = df['step']
    plt.xlabel(args.xlab)
    plt.ylabel(args.ylab)
    legend = []
    if(args.acc):
        y = df['test_acc']
        legend.append("test_acc")
        plt.plot(x,y)
    if(args.loss):
        y = df['test_loss']
        legend.append("test_loss")
        plt.plot(x,y)
    plt.legend(legend)
    if args.savefile != "":
        plt.savefig(args.savefile)
    if args.show:
        plt.show()

