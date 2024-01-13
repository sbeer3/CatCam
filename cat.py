import cv2

def open_camera():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        success = False
        try:
            success, _ = cap.read()
            if success:
                # Set the resolution to 1920x1080
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap  # return the working camera immediately
        except:
            print(f"Failed to open camera at index {index}")
        cap.release()
        index += 1
    return None  # return None if no working camera is found

# Open the camera
import tensorflow as tf
import numpy as np

# Load the model
model = tf.saved_model.load('./model/saved_model')

# Open the camera
cap = open_camera()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to the format expected by the model
    frame_resized = cv2.resize(frame, (320, 320))
    input_tensor = tf.convert_to_tensor(frame_resized)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model on the frame
    output_dict = model(input_tensor)

    # For each detection result
    for i in range(int(output_dict.pop('num_detections'))):
        # print("Detection score:", output_dict['detection_scores'][0][i].numpy())
        # print("Detection class:", output_dict['detection_classes'][0][i].numpy())
        # If the detected object is a cat and the detection score is above 0.5
        if output_dict['detection_classes'][0][i] == 17 and output_dict['detection_scores'][0][i] > 0.5:
            # Draw a box around the detected cat
            box = output_dict['detection_boxes'][0][i].numpy()
            cv2.rectangle(frame, (int(box[1] * frame.shape[1]), int(box[0] * frame.shape[0])), (int(box[3] * frame.shape[1]), int(box[2] * frame.shape[0])), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the VideoCapture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()