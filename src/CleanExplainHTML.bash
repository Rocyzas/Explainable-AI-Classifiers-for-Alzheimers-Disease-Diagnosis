#!/bin/bash

# Check the file is exists or not
if [ -f $filename ]; then
   rm ../ExplainHTML/*
   echo "Files from ExplainHTML removed"
fi
