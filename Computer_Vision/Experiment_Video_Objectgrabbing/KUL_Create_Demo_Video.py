import numpy as np
import cv2

# Function to check if the current frame is between the given time range
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

# Function to display a message on the frame
def display_message(frame, message):
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main() -> None:
    cap = cv2.VideoCapture('new.MOV')
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3) / 3)
    frame_height = int(cap.get(4) / 3)

    frame_width_out = int(frame_width / 1.3)
    frame_height_out = int(frame_height / 1.3)


    fourcc = cv2.VideoWriter_fourcc(*'H264')  # saving output video as .mp4
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width_out, frame_height_out))

    handle_first_video(cap, frame_height, frame_width, out)

    cap.release()
    cap = cv2.VideoCapture('donutvid.MOV')

    handle_second_video(cap, frame_height, frame_width, out)

    cap.release()
    cap = cv2.VideoCapture('rollingball.MOV')

    handle_third_video(cap, frame_height, frame_width, out)

    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print("Windows closed")


# First 40 seconds
def handle_first_video(cap, frame_height, frame_width, out):
    while cap.isOpened():

        if not between(cap, 0, 40000):
            break

        ret, frame = cap.read()

        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            frame = cv2.resize(frame, (int(frame_width), int(frame_height)))

            if between(cap, 0, 4000):
                frame = switch_grayscale(cap, frame)
                pass
            elif between(cap, 4000, 12000):
                frame = blurring(cap, frame)
                pass
            elif between(cap, 12000, 20000):
                frame = object_grabbing(cap, frame)
                pass
            elif between(cap, 20000, 25000):
                frame = visualize_edges(cap, frame)
                pass
            elif between(cap, 25000, 35000):
                frame = visualize_circles(cap, frame)
                pass
            elif between(cap, 35000, 40000):
                frame = object_detection(cap, frame)
                pass

            # write frame that you processed to output
            frame = cv2.resize(frame, (int(frame_width / 1.3), int(frame_height / 1.3)))
            out.write(frame)

            # (optional) display the resulting frame
            # cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break


# Seconds 40 -> 50
def handle_second_video(cap, frame_height, frame_width, out):

    sift = cv2.SIFT_create()

    donutimg = cv2.imread('donut.png')
    donutimg = cv2.rotate(donutimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    donutimg = cv2.resize(donutimg, (int(frame_width), int(frame_height)))
    gray_donut_image = cv2.cvtColor(donutimg, cv2.COLOR_BGR2GRAY)

    keypoints_donut_image, descriptors_donut_image = sift.detectAndCompute(gray_donut_image, None)

    while cap.isOpened():

        if not between(cap, 0, 10000):
            break

        ret, frame = cap.read()

        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            frame = cv2.resize(frame, (int(frame_width), int(frame_height)))

            if between(cap, 0, 10000):
                frame = sift_matching(descriptors_donut_image, donutimg, frame, keypoints_donut_image, sift)
                display_message(frame, "SIFT Matching")
            else:
                break

            # write frame that you processed to output
            frame = cv2.resize(frame, (int(frame_width / 1.3), int(frame_height / 1.3)))
            out.write(frame)

            # (optional) display the resulting frame
            # cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break


# Seconds 50 -> 60
def handle_third_video(cap, frame_height, frame_width, out):

    custom_image = cv2.imread('cars.png')
    custom_image = cv2.resize(custom_image, (int(frame_width), int(frame_height)))

    custom_image2 = cv2.imread('mater.png')
    custom_image2 = cv2.resize(custom_image2, (int(frame_width), int(frame_height)))

    while cap.isOpened():
        if not between(cap, 0, 10000):
            break

        ret, frame = cap.read()

        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            frame = cv2.resize(frame, (int(frame_width), int(frame_height)))

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([15, 50, 50])
            upper_yellow = np.array([40, 255, 255])

            yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
            if between(cap, 0, 2000):
                pass
            elif between(cap , 2000, 6000):
                display_message(frame, "Changing ball in Cars")
                contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Iterate through the contours and draw bounding boxes around the yellow regions
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Replace the detected yellow region with the custom image
                    if w > 0 and h > 0:
                        resized_custom_image = cv2.resize(custom_image, (w, h))
                        frame[y:y + h, x:x + w] = resized_custom_image
                pass
            elif between(cap, 6000, 8000):
                display_message(frame, "Trying to hide the ball")
                # Inpainting
                inpainted_frame = cv2.inpaint(frame, yellow_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
                frame = inpainted_frame

                pass
            elif between(cap, 8000, 10000):
                display_message(frame, "Let Cars come back as Mater")
                contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Iterate through the contours and draw bounding boxes around the yellow regions
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Replace the detected yellow region with the custom image
                    if w > 0 and h > 0:
                        resized_custom_image2 = cv2.resize(custom_image2, (w, h))
                        frame[y:y + h, x:x + w] = resized_custom_image2
                pass
            else:
                break

            # write frame that you processed to output
            frame = cv2.resize(frame, (int(frame_width / 1.3), int(frame_height / 1.3)))
            out.write(frame)

            # (optional) display the resulting frame
            # cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break


def sift_matching(descriptors_donut_image, donutimg, frame, keypoints_donut_image, sift):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)
    # Match descriptors between the image and the frame
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_donut_image, descriptors_frame, k=2)
    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.60 * n.distance:
            good_matches.append(m)
    # Draw matched keypoints on the image and the frame
    matched_image = cv2.drawMatches(donutimg, keypoints_donut_image, frame, keypoints_frame, good_matches, None,
                                    flags=0)
    # Resize the matched image for visualization
    matched_image = cv2.resize(matched_image, (frame.shape[1], frame.shape[0]))
    frame = matched_image
    return frame


def switch_grayscale(cap, frame):
    if between(cap, 1000, 2000):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pass
    elif between(cap, 3000, 4000):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pass
    return frame


def blurring(cap, frame):
    if between(cap, 4000, 8000):
        elapsed_time = int(cap.get(cv2.CAP_PROP_POS_MSEC)) - 4000
        kernel_size = max(1, int(elapsed_time / 200)) * 2 + 1
        frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        display_message(frame, "Gaussian Blur: " + str(kernel_size))
        pass
    elif between(cap, 8000, 12000):
        elapsed_time = int(cap.get(cv2.CAP_PROP_POS_MSEC)) - 8000
        diameter = max(1, int(elapsed_time / 200))  # Adjust diameter
        frame = cv2.bilateralFilter(frame, diameter, 75, 75)
        display_message(frame, "Bilateral Filter: " + str(diameter))
        pass
    return frame


def object_grabbing(cap, frame):
    if between(cap, 12000, 14000):
        frame = grab_object_rgb(frame)
        pass
    if between(cap, 14000, 16000):
        frame = grab_object_hsv(frame)
        pass
    if between(cap, 16000, 20000):
        frame = grab_object_hsv_improved(frame)
        pass

    return frame


def grab_object_rgb(frame):  # grab ball
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lower_color_rgb = np.array([150, 150, 0])  # Lower bound for yellow color in RGB
    upper_color_rgb = np.array([255, 255, 65])  # Upper bound for yellow color in RGB
    # lower_color_rgb = np.array([120, 185, 125])  # Lower bound for green color in RGB
    # pper_color_rgb = np.array([180, 230, 185])  # Upper bound for green color in RGB
    mask_rgb = cv2.inRange(frame_rgb, lower_color_rgb, upper_color_rgb)
    display_message(mask_rgb, "Object grabbing RGB")
    return mask_rgb


def grab_object_hsv(frame):  # grab liberty
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color_hsv = np.array([40, 70, 50])  # Lower bound for green color in HSV
    upper_color_hsv = np.array([90, 255, 255])  # Upper bound for green color in HSV
    # lower_color_hsv = np.array([20, 135, 135])  # Lower bound for yellow color in HSV
    # upper_color_hsv = np.array([30, 255, 255])  # Upper bound for yellow color in HSV
    mask_hsv = cv2.inRange(frame_hsv, lower_color_hsv, upper_color_hsv)
    display_message(mask_hsv, "Object grabbing HSV")
    return mask_hsv


def grab_object_hsv_improved(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color_hsv = np.array([40, 70, 50])  # Lower bound for green color in HSV
    upper_color_hsv = np.array([90, 255, 255])  # Upper bound for green color in HSV
    # lower_color_hsv = np.array([20, 100, 100])  # Lower bound for yellow color in HSV
    # upper_color_hsv = np.array([30, 255, 255])  # Upper bound for yellow color in HSV
    mask_hsv = cv2.inRange(frame_hsv, lower_color_hsv, upper_color_hsv)

    # Apply morphological operations to enhance mask
    kernel = np.ones((5, 5), np.uint8)

    # Remove small noise pixels
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    # Fill in small gaps and holes
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
    # Create a copy of the original mask to overlay improvements
    mask_hsv_overlay = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
    mask_hsv_overlay[np.where((mask_hsv_overlay == [255, 255, 255]).all(axis=2))] = [0, 0,
                                                                                     255]  # Overlay improvements in red

    # Overlay the improved mask on the original mask
    overlay = cv2.addWeighted(cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR), 0.5, mask_hsv_overlay, 0.5, 0)

    display_message(overlay, "Object grabbing HSV + Morphological Operations (Open & Close)")

    return overlay


def visualize_edges(cap, frame):
    message = ""
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if between(cap, 20000, 21000):
        message = "Sobel horizontal edges | kernel size=3 | derivative=1"
        horiz_edges = return_horizontal_edges(frame, gray)
        display_message(horiz_edges, message)
        return horiz_edges
    elif between(cap, 21000, 22000):
        message = "Sobel vertical edges | kernel size=3 | derivative=1"
        vertic_edges = return_vertical_edges(frame, gray)
        display_message(vertic_edges, message)
        return vertic_edges
    else:
        # 0 - 5000
        ksize = 3
        der = 1
        message = "Sobel vertic & horiz edges | kernel size=3 | derivative=1"
        if between(cap, 23000, 24000):
            message = "Sobel vertic & horiz edges | kernel size=1 | derivative=1"
            ksize = 1
            pass
        elif between(cap, 24000, 25000):
            message = "Sobel vertic & horiz edges | kernel size=3 | derivative=2"
            der = 2
            pass

        # Apply Sobel operator for horizontal edges
        sobel_horizontal = cv2.Sobel(gray, cv2.CV_64F, der, 0, ksize=ksize)
        # Apply Sobel operator for vertical edges
        sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, der, ksize=ksize)

        abs_horizontal = cv2.convertScaleAbs(sobel_horizontal)
        abs_vertical = cv2.convertScaleAbs(sobel_vertical)

        # Create masks for horizontal and vertical edges
        mask_horizontal = np.zeros_like(gray)
        mask_horizontal[abs_horizontal > 100] = 255  # Threshold can be adjusted
        mask_vertical = np.zeros_like(gray)
        mask_vertical[abs_vertical > 100] = 255  # Threshold can be adjusted

        # Create red color for horizontal edges and green color for vertical edges
        colored_edges = np.zeros_like(frame)
        colored_edges[:, :, 2] = mask_horizontal  # Red channel for horizontal edges
        colored_edges[:, :, 1] = mask_vertical  # Green channel for vertical edges

        display_message(colored_edges, message)

        return colored_edges


def return_vertical_edges(frame, gray):
    # Apply Sobel operator for vertical edges
    sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_vertical = cv2.convertScaleAbs(sobel_vertical)
    mask_vertical = np.zeros_like(gray)
    mask_vertical[abs_vertical > 100] = 255  # Threshold can be adjusted
    colored_edges = np.zeros_like(frame)
    colored_edges[:, :, 1] = mask_vertical  # Green channel for vertical edges
    return colored_edges


def return_horizontal_edges(frame, gray):
    # Apply Sobel operator for horizontal edges
    sobel_horizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_horizontal = cv2.convertScaleAbs(sobel_horizontal)
    mask_horizontal = np.zeros_like(gray)
    mask_horizontal[abs_horizontal > 100] = 255  # Threshold can be adjusted
    colored_edges = np.zeros_like(frame)
    colored_edges[:, :, 2] = mask_horizontal  # Red channel for horizontal edges
    return colored_edges


def visualize_circles(cap, frame):
    minDist = 50

    param1 = 50
    param2 = 50
    min_radius = 0
    max_radius = 0

    message = ""

    if between(cap, 25000, 27000):
        message = "Hough Circles | param1=50 | param2=30"
        param1 = 50
        param2 = 40
    elif between(cap, 27000, 29000):
        message = "Hough Circles | param1=50 | param2=50"
        param1 = 50
        param2 = 50
    elif between(cap, 29000, 31000):
        message = "Hough Circles | param1=30 | param2=50"
        param1 = 40
        param2 = 50
    elif between(cap, 31000, 33000):
        message = "Hough Circles | param1=50 | param2=50"
        param1 = 50
        param2 = 50
    elif between(cap, 33000, 35000):
        message = "Hough Circles | param1=50 | param2=50 | radius=20 -> 50"
        param1 = 50
        param2 = 50
        min_radius = 20
        max_radius = 50

    display_message(frame, message)

    # Convert the yellow only mask to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (10, 10))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=minDist,  # Adjust as per your requirement
        param1=param1,  # Adjust as per your requirement
        param2=param2,  # Adjust as per your requirement
        minRadius=min_radius,  # Adjust as per your requirement
        maxRadius=max_radius  # Adjust as per your requirement
    )

    # Draw circles that are detected.q
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(frame, (a, b), r, (0, 255, 0), 3)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)

    return frame


def object_detection(cap, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CODE USED TO CUT OUT THE TEMPLATE
    # roi_top_left = (975, 400)  # Example coordinates of the top-left corner of the ROI
    # roi_bottom_right = (1050, 520)  # Example coordinates of the bottom-right corner of the ROI
    #
    # # Slice the ROI out of the frame
    # roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    template = cv2.imread('ROI.png')
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template.shape[0], template.shape[1]

    matched = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)

    if between(cap, 35000, 37000):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
        return frame
    elif between(cap, 37000, 40000):
        normalized_res = cv2.normalize(matched, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        normalized_res_resized = cv2.resize(normalized_res, (frame.shape[1], frame.shape[0]))
        return normalized_res_resized


if __name__ == '__main__':
    main()
