#!/bin/bash -e

PASSED() {
  printf "\t\33[32m[Passed]\33[0m\n"
}

FAILED() {
  printf "\t\33[31m[Failed]\33[0m See $LOG for more detail.\n"
}

RUN_TEST() {
  l=$1
  LOG=.level-${l}.log
  printf "Running level $l test ..."
  ( (./level-${l}.sh > $LOG 2>&1) && PASSED) || FAILED $LOG
}

RUN_TEST 0
RUN_TEST 1
RUN_TEST 2
RUN_TEST 3
