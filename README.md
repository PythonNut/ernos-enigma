## What?
Hold a Rubik's cube in front of a camera. Our amazing software will help you along, and you will find yourself solving the cube in NO TIME!

## Why?
You can look cool to other people. Maybe

## How?
Use OpenCV to locate the Rubik's cube within the image by detecting edges, then
identifying enclosed regions of the appropriate size. After that, we use an algorithm based on space partitioning trees to determine the set of regions that belongs to the cube. Then create a distance metric on the HSV
values of each sticker to identify the color of each sticker.

Send the sticker information to another process with sockets, and generate a simple series of
moves that will solve the Rubik's cube.

Finally, convert that series of moves into an interactive experience using pygame.

## Challenges
Lots of problems with everything (detecting edges, stickers, matching colors, getting lighting correct, communicating with sockets, getting pygame working (Windows is sad)), but we got past some of them. And faked the rest.

## Other thoughts
It worked. It's actually incredible.


