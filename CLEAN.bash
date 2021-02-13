#!/bin/bash

# Check the file is exists or not
if [ -f $filename ]; then
   rm ExplainHTML/* Models/*
   echo "Files from Models and ExplainHTML removed"
fi
