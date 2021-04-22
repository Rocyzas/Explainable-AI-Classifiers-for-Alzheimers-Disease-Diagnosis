from Train import trainEachClassifier
import time
import sys

if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 'classification_method', 0 or 1 save
    trainEachClassifier(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
