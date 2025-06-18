import os

# Update these paths to your actual folders
IMG_DIR = r"C:\coding\minor project\AI ENABLED SYSTEM FOR ROAD SIGN\yolov5_data\images\val"
LBL_DIR = r"C:\coding\minor project\AI ENABLED SYSTEM FOR ROAD SIGN\yolov5_data\labels\val"

# They go from 00500–00599; we want to remap 00500→00000, 00501→00001, …, 00599→00099
START = 500
COUNT = 100

for i in range(START, START + COUNT):
    old_stem = f"{i:05d}"
    new_stem = f"{i - START:05d}"

    # Rename image
    old_img = os.path.join(IMG_DIR,  old_stem + ".jpg")
    new_img = os.path.join(IMG_DIR,  new_stem + ".jpg")
    if os.path.exists(old_img):
        os.rename(old_img, new_img)
    else:
        print(f"Image not found: {old_img}")

    # Rename label
    old_lbl = os.path.join(LBL_DIR,  old_stem + ".txt")
    new_lbl = os.path.join(LBL_DIR,  new_stem + ".txt")
    if os.path.exists(old_lbl):
        os.rename(old_lbl, new_lbl)
    else:
        print(f"Label not found: {old_lbl}")
