# logic-simplifier
A (possibly working?) simplifier for logical formulas...

## Some Background

This is an attempt at an exercise from a discrete maths course I'm taking. The
task was to (manually) simplify a logical formula in the least number of steps
possible using only a limited number of allowed equivalence transformations.
That was a lot of trial and error, so I wrote this brute force simplifier.
Unfortunately, it wasn't performant enough and I gave up eventually.

Also...
 - there are no tests
 - there's probably a loooot of bugs
 - there is no documentation
 - I'm not sure if the program is even working in the current state

## Implementation

If I remember correctly, the formulas are represented as binary trees and stored
inside vectors in prefix notation.

The equivalence transformations are applied using a pattern matching implementation.
Transformations are defined as a matching pattern and a replacement pattern. At each
node of the depth-first search, the formula is matched against all allowed transformation
and each match is then replaced by the transformation's replacement pattern. The resulting
formulas are the child nodes of the current one. Rinse and repeat... until... maybe (it's
not very likely though) a solution is found.
