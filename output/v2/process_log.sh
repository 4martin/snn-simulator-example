#!/bin/sh

cat log.1 | grep ^fired | uniq | sort -t" " -k3 > log.1.fired
cat log.2 | grep ^fired | uniq | sort -t" " -k3 > log.2.fired
