# ImageStitching-
This was an assignment for a computer vision class. The assignment was to stich to images together, 
however it was as easy as feeding in the resulting image back into the loop to stich multiple images together.
As it is in its current state, images compared together must be overlapping somewhere for it to work properly, 
and if two related images are compared, with no overlap then the final result will be an error

    eg:
    A overlaps with B
    C overlaps with B
    A and C have no overlap
    
    if we feed in (A,B) = AB, and (AB,C) = ABC, we will get a correct image
    however, (A,C) = AC and then (AC,B) = ACB, the result will probably be incorrect
