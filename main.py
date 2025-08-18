import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time
import numpy as np
import os
import rasterio
import cv2
import skimage
from shapely import wkt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rasterio.plot import show
import json


#TODO
# *Clean up code
# *Remove uncessary libraries
# *Fetch camera size
# *Add Final functioality with x/y coords

model = YOLO("yolo11s.pt")
cap = cv2.VideoCapture(0)
target_class = 'person'


def junt(coords, screensize=0):
    # fig = figure_object, ax = axes_object for indexing subplots
    fig, ax = plt.subplots(dpi=100, figsize=(4.8, 6.4), )
    fig.set_facecolor("#000000")  # default is 255/ #FFFFFF which causes an error

    # For every feature get the class and coordinates for plotting

    # should look like: "POLYGON ((-99.22825896166003 19.3070188729944, -99.22792409605501 19.30722016299979, -99.22783779973361 19.30709761467095, -99.2281745103911 19.3068767686493, -99.22825896166003 19.3070188729944))"
    wkt_str = (f"POLYGON (({x1} {y1}, {x2} {y1}, {x2} {y2}, {x1} {y2}, {x1} {y1}))")
    polygon = wkt.loads(wkt_str)
    coords = [(x, y) for x, y in polygon.exterior.coords]

    # Use rasterio to show the image with correct orientation
    show(np.zeros((480, 640)), ax=ax, cmap='gray')

    # Create patch with color handling from above
    patch = patches.Polygon(coords, closed=True, edgecolor="white", facecolor="white", fill=False,
                            linewidth=1, aa=False, rasterized=True)

    ax.add_patch(patch)



    """
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding
    plt.savefig("buffer.png", bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.show()
    plt.close(fig)
    """
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()
    plt.close(fig)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(.5)
    # Run inference
    results = model(frame)

    # Annotate frame
    for result in results:
        boxes = result.boxes
        annotator = Annotator(frame)
        result.save("hamburger_with_extra_cheese.jpg")
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]

            # Filter for a class
            if label != target_class:
                continue

            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box and label
            annotator.box_label([x1, y1, x2, y2], f"{label} {conf:.2f}")

            # Print (or return) bbox data
            print({
                "bbox": [x1, y1, x2, y2],
                "class": label,
                "confidence": conf
            })
            if conf > .7:
                junt(coords=[x1, y1, x2, y2])

        # Annotate the frame
        frame = annotator.result()

    # Show the frame
    cv2.imshow("Real-time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
