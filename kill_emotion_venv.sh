#!/usr/bin/env bash

VENVNAME=emotion
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME