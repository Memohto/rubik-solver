import cv2 as cv
import numpy as np
from rubik_solver import utils

color_offset = 70
blue = np.array([30,80,130])
green = np.array([40,120,70])
orange = np.array([220,69,70])
red = np.array([150,30,50])
white = np.array([30,30,40])
yellow = np.array([160,160,80])

def predict_color(color):
    if all(l <= c <= h for c, l, h in zip(color, blue - color_offset, blue + color_offset)):
        return 'b'
    elif all(l <= c <= h for c, l, h in zip(color, green - color_offset, green + color_offset)):
        return 'g'
    elif all(l <= c <= h for c, l, h in zip(color, orange - color_offset, orange + color_offset)):
        return 'o'
    elif all(l <= c <= h for c, l, h in zip(color, red - color_offset, red + color_offset)):
        return 'r'
    elif all(l <= c <= h for c, l, h in zip(color, white - color_offset, white + color_offset)):
        return 'w'
    elif all(l <= c <= h for c, l, h in zip(color, yellow - color_offset, yellow + color_offset)):
        return 'y'
    return '-'


cap = cv.VideoCapture(0)

cube_dic = {
    0: "UP",
    1: "LEFT",
    2: "FRONT",
    3: "RIGHT",
    4: "BACK",
    5: "DOWN"
}
cube_strings = ['', '', '', '', '', '']

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    gray_frame = cv.GaussianBlur(gray_frame, (5,5), 50, 50)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(3, 3))

    canny = cv.Canny(gray_frame, 20, 60)
    dilatation = cv.dilate(canny, kernel, iterations=5)

    flooded_frame = dilatation.copy()
    h, w = flooded_frame.shape[:2]
    offset = 10

    cv.floodFill(flooded_frame, None, (offset, offset), 255)
    cv.floodFill(flooded_frame, None, (offset, h - offset), 255)
    cv.floodFill(flooded_frame, None, (w - offset, h - offset), 255)
    cv.floodFill(flooded_frame, None, (offset, h - offset), 255)

    cube_mask = cv.bitwise_not(flooded_frame)

    cube_mask = cv.dilate(cube_mask, kernel, iterations=60)
    cube_mask = cv.erode(cube_mask, kernel, iterations=50)

    contours, hierarchy = cv.findContours(cube_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    current_string = ''
    if len(contours) > 0:
        x,y,w,h = cv.boundingRect(contours[0])
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)

        n_cells = 3
        cell_width = w // n_cells
        cell_height = h // n_cells
        for i in range(n_cells):
            y_current = i * cell_height
            for j in range(n_cells):
                x_current = j * cell_width

                x_pos = x + x_current
                y_pos = y + y_current
                cv.rectangle(frame,(x_pos,y_pos),(x_pos+cell_width,y_pos+cell_height),(0,255,0),3)

                window = rgb_frame[y_pos:y_pos + cell_height, x_pos:x_pos + cell_width]
                center_pixel = window[window.shape[0]//2, window.shape[1]//2]
                # print(center_pixel)
                letter = predict_color(center_pixel)
                current_string += letter
                frame = cv.putText(frame, letter, (x_pos+cell_height//3, y_pos+cell_height//2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # print("")

    for i in range(len(cube_strings)):
        frame = cv.putText(frame, cube_dic[i]+": "+cube_strings[i], (10, 25+i*25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv.imshow('Result', frame)
    key = cv.waitKey(1)
    if key == ord('e'):
        cv.destroyAllWindows()
        break
    if key == ord('u'):
        print("UP")
        cube_strings[0] = current_string
    if key == ord('l'): 
        print("LEFT")
        cube_strings[1] = current_string
    if key == ord('f'): 
        print("FRONT")
        cube_strings[2] = current_string
    if key == ord('r'): 
        print("RIGHT")
        cube_strings[3] = current_string
    if key == ord('b'): 
        print("BACK")
        cube_strings[4] = current_string
    if key == ord('d'): 
        print("DOWN")
        cube_strings[5] = current_string
    if key == ord('s'):
        print("SOLVE")
        final_string = "".join(cube_strings)
        print(final_string)
        try:
            print(utils.solve(final_string, 'Kociemba'))
        except:
            print("Could not solve cube, try again")

cv.destroyAllWindows()