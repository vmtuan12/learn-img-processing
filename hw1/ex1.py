import cv2 
import matplotlib.pyplot as plt

# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    return cv2.imread(image_path)

# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Remember to use plt.show() to display the image
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis(False)
    plt.show()

# Grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)

# Flip an image as function
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    return cv2.flip(image, 1)

# Rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    # get image size
    (width, height) = image.shape[1::-1]
    # get image center
    image_center = (width/2, height/2) 

    rotate_matrix = cv2.getRotationMatrix2D(center=image_center, angle=angle, scale=1)
    result = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return result


if __name__ == "__main__":
    # Load an image from file
    img = load_image("uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
