# PennyLane coding challenges, QHack 2026

My team PewPewQew participated in the Qhack challenge this fall. Very fun, very challenging problems. These are
the six I worked on, documented in the form: problem, solution and the approach.

I have included the solution and the figures for each of the individual problems in the folders.

The last folder is a bonus problem, where I did some extended work on an open ended research problem disguised as a challenge. Kudos to the teams who were able to successfully submit working solutions to that problem, IN 24 HOURS. Given that ITS LITERALLY AN OPEN PROBLEM.

| | Challenge | Result | Approach (WHYYYYY?) |
|---|---|---|---|
| 1 | [Certifying a GHZ state](01-ghz-parity-checks/) | score 1.0, with exactly the n required checks | X parity sees the coherence, ZZ parities pin the populations |
| 2 | [Grover with half the list marked](02-grover-half-the-list/) | P(success) = 1 for all six oracles, one shot | a controlled oracle turns M/N = 1/2 into 1/4, which one step solves exactly |
| 3 | [QAOA approximation ratio](03-qaoa-approximation-ratio/) | matches the reference to 10⁻⁸ | the ratio describes one optimization run, so I had to find that run |
| 4 | [QPE in a time budget](04-qpe-resource-budget/) | both public cases exact | one more bit of phase costs twice the T gates |
| 5 | [Breaking the tie in amplitude estimation](05-break-the-tie/) | matches to the last digit | a cheap direct circuit picks between a and 1 − a |
| 6 | [Unitary synthesis](06-unitary-synthesis/) | Silver, 21.06% below the Shannon decomposition | exactly 4ⁿ − 1 rotations, which is the floor |
| 7 | [How many T gates does a unitary need?](07-how-many-t-gates/) | Gold needs whole-unitary Clifford+T synthesis | counting says the circuits exist; nobody knows how to find them |

## What I took from it

- **Check every stage against something exact.** Each solution here is tested against a closed form, a
  known group size, or every possible input, not only the public test cases. That is how I caught a
  grader that passes broken circuits in challenge 2.
- **A grader measures its own procedure.** In challenge 3 the "right" answer was a local minimum reached by
  one specific optimizer. Optimizing better would have failed.
- **Count the parameters before writing code.** It explained why Grover gets stuck at M = N/2, why my
  synthesis circuit cannot be beaten with rotations, and why Gold needs a different kind of circuit.

## How I worked

During the event I used Claude, Anthropic's AI model, as a tutor. It explained the physics, checked my
reasoning and reviewed my code, and I wrote the submitted solutions myself, which QHack's staff confirmed
was within the rules. After the event I went back through everything with Claude to verify each solution,
make the figures and write these notes. The T-count study in part 7 was done together with it.

The problem statements are summarized in my own words, and the original challenges are Xanadu's, written for
QHack 2026. Where a challenge came with helper code, I rebuilt what I needed from its description, so
nothing here is copied from the challenge templates.

## Run

```
pip install -r requirements.txt
cd 06-unitary-synthesis && python solution.py && python figures.py
```

Every folder works the same way: `python solution.py` runs the checks, and `python figures.py` redraws the
figures. Tested with Python 3.13 and PennyLane 0.45.
