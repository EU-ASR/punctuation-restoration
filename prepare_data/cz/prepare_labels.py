#!/bin/env python3

import sys, os, re

# define the symbols

kPunctMap = { "." : "PERIOD", "," : "COMMA", "!" : "EXCLAMATION", "?" : "QUESTION" }
kNoPunct = "O"

kLowerCase = "O"
kTitleCase = "TitleCase"
kAllCaps = "ALL_CAPS"


def prepare_labels(fd):
    """
        Prepare labels for punctuation model training.
    """

    while True:
        line = fd.readline()
        if not line: break
        line = line.strip()

        tokens = line.split()

        for ii,wrd in enumerate(tokens):
            # skip punctuation
            if wrd in kPunctMap.keys():
                continue

            ### CASE_TYPE
            # get the 'case type' of the word:
            wrd_ = re.sub(r"[/-]", "", wrd)
            if wrd_.isupper():
                case_type = kAllCaps
            elif wrd_.istitle():
                case_type = kTitleCase
            else:
                case_type = kLowerCase

            ### PUNCTUATION
            # look-up following token,
            # see if it is punctuation
            punct_type = kNoPunct # default
            if ii+1 < len(tokens):
                next_token = tokens[ii+1]
                if next_token in kPunctMap.keys():
                    punct_type = kPunctMap[next_token]

            ### OUTPUT
            wrd_lc = wrd.casefold() # ~lower()
            print(f"{wrd_lc}\t{case_type},{punct_type}")



if __name__ == "__main__":
    prepare_labels(sys.stdin)

