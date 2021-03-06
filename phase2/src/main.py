from trainNavigation import navigation
import time
import sys

if __name__ == '__main__':
    start_time = time.time()

    currentProj = ["mean", "max", "std"]
    # typeExclude = ["multi", "AD", "MCI", "CN"]
    typeExclude = ["multi"]

    # for each projection and type of classification
    for proj in currentProj:
        for typ in typeExclude:
            navigation(proj, typ)

    print("--- %.2f seconds ---" % (time.time() - start_time))
