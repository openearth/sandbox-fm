# Virtual-River-prototype
Initial script to detect Virtual River game status and process the information in a way that can be used as input by the SandBox framework

The script covers a calibration and image processing script to analyze the current game state of Virtual River, based on the SandBox framework. It processes pictures taken of the current game board, taken from below the game board, and analyzes the game pieces that are placed in each gridcell based on markers.

Two versions of the script are added to the repository. The normal and elaborate versions. Both perform the same functions, and the normal should be used to work from. The elaborate version of the functions stores intermediate steps to help show how the script works.

First, it detects to outside corners of the grid and warps the picture to match those corners as corners of the picture. This perspective warps is stored as the calibration of the picture. It can be performed before a game session, be repeated during game session or be included in the game state analysis (calibration, on my laptop, takes roughly 1 second). Next, the calibration is used to also warp the image that needs processing. With the perspective warp performed, the locations of each grid cell are known - with perhaps a minor offset from the image processoing. The image is first turned into a HSV color range to better distinguish between the red (geometry) and blue markers (ecotopes). Two masks are created that only pass through the red and blue makers respectively as white, turning the rest of the image black. A for loop is subsequently used to cycle through the locations of each grid cell as the region of interest. Here, a circular mask is applied to remove any shapes from adjacent grid cells and the number of counters - correponding to the amount of coloured squares) are counted. The amount of contours identified for each grid cell is then stored in a numpy array with three columns:

●	grid cell number (from 1 to 143)

●	geometry height (0, 1, 2, 3 or 4) - 0 in the example corresponds to river bed level, 2 to floodplain level and 4 to dike level

●	aggregated ecotope type (1: grassland, 2: brushwood, 3: thicket, 4: forest, 5: unvegetated, 6: agriculture, 7: build structure,
  8: river bed/water) - these are preliminary

The filename variable in the Processimage or Processimage_elaborate can be changed from 'marker4.jpg' to 1, 2 or 3 to process another sample image.

In case the calibration is combined with image processing (most robust option), the code could be made more efficient as both the calibration image and processing image are warped. It's also an option to periodically (while players are making changes) renew, calibrate and possibly process pictures on fixed intervals or when the facilitator initiates it.
