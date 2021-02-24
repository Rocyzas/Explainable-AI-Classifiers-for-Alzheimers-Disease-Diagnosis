#!/bin/bash

# Check the file is exists or not
if [ -f $filename ]; then
   rm -rf ../ExplainHTML/*
   echo "Files from ExplainHTML removed"
fi
