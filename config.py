import argparse

def read_args():

    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('-j','--jobs', type=int, help='jobs', required=True)
    parser.add_argument('-m','--macs', type=int, help='machines', required=True)
    parser.add_argument('-t','--maxTime', type=int, help='maximum task time', required=True)


    args = vars(parser.parse_args())

    return args