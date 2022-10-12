#!/usr/bin/env python3

import os
import sys

if len(sys.argv) != 2:
  print("Usage: ./gen_run.py <number of testcases to be used in run command>")
  exit(1)

testcases = os.listdir(os.getcwd() + '/tests')
num_testcases = min(int(sys.argv[1]), len(testcases))

result = "./scanner signatures/sigs-exact.txt"
for testcase in testcases[:num_testcases]:
  result += " tests/" + testcase

print(result)
