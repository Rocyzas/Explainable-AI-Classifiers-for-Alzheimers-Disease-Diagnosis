#!/bin/bash

# Check the file is exists or not
if [ -f $filename ]; then
   rm -rf ../Models/*.joblib ../ExplainHTML/* ../Models/FeatureImportance
   echo "All files from Models and ExplainHTML removed"
fi
