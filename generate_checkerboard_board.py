import numpy as np
import cv2


def generate_checkerboard(width, height, square_size):
    # Create checkerboard pattern
    checkerboard = np.zeros((height * square_size, width * square_size), dtype=np.uint8) + 120
    for j in range(8,  8+13):
        for i in range(2, 2+13):
            if (i + j) % 2:
                checkerboard[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size] = 255
            else:
                checkerboard[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size] = 0

    return checkerboard


# Checkerboard settings
width = 32  # number of squares along the width
height = 18  # number of squares along the height
square_size = 80  # size of each square in pixels
screen_height = 33.622
screen_width = 59.772

# Generate checkerboard
checkerboard = generate_checkerboard(width, height, square_size)

# Save or display the image
# cv2.imwrite('checkerboard.png', checkerboard)
cv2.imshow('Checkerboard', checkerboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
