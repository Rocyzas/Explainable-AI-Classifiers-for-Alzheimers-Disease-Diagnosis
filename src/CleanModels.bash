#!/bin/bash

# Check the file is exists or not
if [ -f $filename ]; then
   rm ../Models/*
   echo "Files from Models removed"
fi
