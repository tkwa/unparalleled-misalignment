An in-progress tool for generating [unparalleled misalignments](https://rickiheicklen.com/unparalleled-misalignments.html).

Current goal is to score a candidate phrase pair "a0 a1 // b0 b1" as something like:

    score(a, b) = synonym_score(a0, b0) + synonym_score(a1, b1) + \
                  collocation_score(a) + collocation_score(b) + \
                  dramatic_meaning_difference(a, b)