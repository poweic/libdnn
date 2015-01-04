#!/bin/bash -e

PASSED() {
  printf "\t\33[32m[Passed]\33[0m\n"
}

FAILED() {
  printf "\t\33[31m[Failed]\33[0m\n"
}

RUN_TEST() {
  l=$1
  printf "Running level $l test ..."
  ( (./level-${l}.sh > /dev/null 2>&1) && PASSED) || FAILED
}

RUN_TEST 0
RUN_TEST 1
RUN_TEST 2
