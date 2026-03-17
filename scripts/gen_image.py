from PIL import Image, ImageDraw

# # Create a simple colored image
# img = Image.new("RGB", (256, 256), color="skyblue")

# # Draw a red square in the middle
# draw = ImageDraw.Draw(img)
# draw.rectangle([80, 80, 180, 180], fill="red")

# img.save("input.jpg")
# img.show()
mask = Image.new("L", (256, 256), color=0)  # black mask
draw = ImageDraw.Draw(mask)
draw.rectangle([80, 80, 180, 180], fill=255)  # white area over the red square

mask.save("mask.png")
mask.show()
