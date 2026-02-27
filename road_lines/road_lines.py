import cv2 as cv
import numpy as np

def main():
    img = cv.imread('photos/road/Screenshot 2026-02-05 170250.png')

    img_pr = process_image(img)
    print(img.shape[1], img.shape[0])

    lines = find_lines(img_pr)
    draw_lines(img, lines)

    cv.imshow('lines', img)

    cv.waitKey(0)
    cv.destroyAllWindows()
    

def process_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    low_t = 10
    high_t = 150
    edges = cv.Canny(blur, low_t, high_t)

    masked_edges = create_mask(edges)

    return masked_edges


def create_mask(img):
    vertecies = np.array([
        [
            (0, img.shape[0]),
            (260, 180),
            (300, 180),
            (img.shape[1], img.shape[0])
        ]
    ], dtype=np.int32)

    mask = np.zeros_like(img)
    ignore_mask_color = 255
    cv.fillPoly(mask, vertecies, ignore_mask_color)

    masked_edges = cv.bitwise_and(img, mask)

    return masked_edges


def find_lines(img):
    rho = 3
    theta = np.pi / 180
    threshold = 15
    min_line_len = 60
    max_line_gap = 150

    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    return lines


def draw_lines(img, lines, color=(75,75, 0), thickness=2):
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []

    y_bottom = img.shape[0]
    y_upper = 170

    for line in lines:
        for (x1, y1, x2, y2) in line:
            if x1 == x2:
                continue
            elif ((y2 - y1) / (x2 - x1)) > 0.3 and ((y2 - y1) / (x2 - x1)) < 0.8:
                slope = (y2 - y1) / (x2 - x1)
                b = y1 - slope * x1
    
                x_bottom_pos.append((y_bottom - b) / slope)
                x_upper_pos.append((y_upper - b) / slope)

            elif ((y2 - y1) / (x2 - x1)) < -0.3 and ((y2 - y1) / (x2 - x1)) > -0.8:
                slope = (y2 - y1) / (x2 - x1)
                b = y1 - slope * x1

                x_bottom_neg.append((y_bottom - b) / slope)
                x_upper_neg.append((y_upper - b) / slope)

    lines_mean = np.array(
        [
            [int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper))],
            [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper))]
        ]
    )

    for i in range(len(lines_mean)):
        cv.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)


if __name__ == '__main__':
    main()