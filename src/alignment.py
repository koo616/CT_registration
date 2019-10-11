import cv2
import numpy as np
from scipy import io
import nibabel as nib
import os

pre_len = []
del_len = []
for i in range(1, 301):
    imgs = io.loadmat("../../dataset/urinary/vol_{}.mat".format(i))
    pre = imgs['imgV_pre']
    delay = imgs['imgV_delay']

    if pre.shape[0] != 512 or pre.shape[1] != 512 or delay.shape[0] != 512 or delay.shape[1] != 512:
        print("width, height doesn't match! {}".format(i))
        print(pre.shape)
        print(delay.shape)

    # pre_nib = nib.Nifti1Image(pre, affine=np.eye(4))
    # delay_nib = nib.Nifti1Image(delay, affine=np.eye(4))
    # nib.save(pre_nib, os.path.join('result', "pre{}.nii.gz".format(i)))
    # nib.save(delay_nib, os.path.join('result', "delay{}.nii.gz".format(i)))

cv2.imwrite("113_pre.png", (im[:, :] / 4096 * 255).astype(np.uint8))
cv2.imwrite("113_delay.png", (imRef[:, :] / 4096 * 255).astype(np.uint8))


pre = (pre / 4096) * 255

pre.max()

imgs = io.loadmat("../../dataset/urinary/vol_{}.mat".format(113))
im = imgs['imgV_pre'][:, :, 0]
imRef = imgs['imgV_delay'][:, :, 0]



warp_img = nib.Nifti1Image(pre, affine=np.eye(4))
nib.save(warp_img, os.path.join('result', "pre228.nii.gz"))


MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.5

imgs = io.loadmat("../../dataset/urinary/vol_1.mat")
imgs.keys()
pre = imgs['imgV_pre']
delay = imgs['imgV_delay']

files = np.load("data/train/E0000325.npz")
pre = files['nojoyoungje']
delay = files['joyoungje']
pre_part = pre[:, :100, 0]
delay_part = delay[:, :100, 0]
#cv2.imwrite("result/testimg.png", pre_part)
# a = cv2.imread("result/testimg.png")

pre = 255 * (pre + 1024) / 4096
pre = pre.astype(np.uint8)

pre_part = 255 * (pre_part + 1024) / 4096
pre_part = pre_part.astype(np.uint8)
delay_part = 255 * (delay_part + 1024) / 4096
delay_part = delay_part.astype(np.uint8)

orb = cv2.ORB_create(MAX_MATCHES)
keypoints1, descriptors1 = orb.detectAndCompute(pre_part, None)
keypoints2, descriptors2 = orb.detectAndCompute(delay_part, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imMatches = cv2.drawMatches(pre_part, keypoints1, delay_part, keypoints2, matches, None)
cv2.imwrite("result/matchimg.png", imMatches)

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)


for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt


# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
points1
points2
# Use homography
height, width, c = delay.shape
im1Reg = cv2.warpPerspective(pre[:, :, 0], h, (width, height))

cv2.imwrite("result/aligned.png", im1Reg)

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def align_mat(im, imRef):
    # im and imRef is 2d-numpy array
    # im & imRef's dtype is uint8 & range is 0~255
    # patient_mat = np.load('./data/test/E0000325.npz')
    #
    # # Convert images to grayscale
    # im = patient_mat['nojoyoungje'][:,:,0]
    # imRef = patient_mat['joyoungje'][:,:,0]
    imPart = im[:100, :]
    imRefPart = imRef[:100, :]
    # im = im[:, :, np.newaxis]
    # imRef = imRef[:, :, np.newaxis]
    imPart = imPart[:, :, np.newaxis]
    imRefPart = imRefPart[:, :, np.newaxis]
    # im = (im + 1024) / 4096 * 255
    # imRef = (imRef + 1024) / 4096 * 255
    # imPart = (imPart + 1024) / 4096 * 255
    # imRefPart = (imRefPart + 1024) / 4096 * 255
    # im= im.astype('uint8')
    # imRef = imRef.astype('uint8')
    # imPart= imPart.astype('uint8')
    # imRefPart = imRefPart.astype('uint8')

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(imPart, None)
    kpRef, desRef = orb.detectAndCompute(imRefPart, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des, desRef, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = 4
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(imPart, kp, imRefPart, kpRef, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp[match.queryIdx].pt
        points2[i, :] = kpRef[match.trainIdx].pt

    # Find homography
    h = cv2.getPerspectiveTransform(points1, points2)

    # Use homography
    # height, width, channels = imRef.shape
    # im1Reg = cv2.warpPerspective(im, h, (width, height))

    return h


if __name__ == '__main__':
    # Read reference image
    refFilename = "form.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "scanned-form.jpg"
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
