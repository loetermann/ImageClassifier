/*
 * Copyright (C) 2016 Lars Oetermann <lars.oetermann.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.oetermann.imageclassifier;

import java.util.List;
import java.util.ListIterator;
import org.bytedeco.javacpp.opencv_core.KeyPointVector;
import org.bytedeco.javacpp.opencv_core.KeyPointVectorVector;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_features2d.Feature2D;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_xfeatures2d;

/**
 *
 * @author Lars Oetermann <lars.oetermann.com>
 */
public class DescriptorExtractorWrapper {

    private final Feature2D feature2D;
    private final DetectorType detectorType;

    public DescriptorExtractorWrapper(DetectorType detectorType) {
        switch (detectorType) {
            case SURF:
                feature2D = opencv_xfeatures2d.SURF.create();
                break;
            case ORB:
                feature2D = opencv_features2d.ORB.create();
                break;
            default:
                throw new IllegalArgumentException("Unsupported detector type: " + detectorType);
        }
        this.detectorType = detectorType;
    }

    public DescriptorExtractorWrapper(String detectorType) {
        this(DetectorType.valueOf(detectorType));
    }

    public DescriptorExtractorWrapper() {
        this(DetectorType.ORB);
    }

    public MatVector readImages(List<String> files, boolean grayscale) {
        MatVector images = new MatVector(files.size());
        Mat mat;

        long i = 0;
        for (ListIterator<String> it = files.listIterator(); it.hasNext();) {
            String file = it.next();

            mat = opencv_imgcodecs.imread(file);
            if (mat.dims() > 0 && mat.cols() > 0 && mat.rows() > 0) {
                if (grayscale) {
                    opencv_imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_RGB2GRAY);
                }
                images.put(i, mat);
                i++;
            } else {
                it.remove();
                System.out.println("Cannot read file: " + file);
            }
        }
        images.resize(i);
        return images;
    }

    public MatVector detectAndCompute(List<String> images, boolean grayscale) {
        MatVector imageList = readImages(images, grayscale);
        MatVector descriptors = detectAndCompute(imageList);
//        imageList.stream().forEach((image) -> {
//            image.release();
//        });
        return descriptors;
    }

    public MatVector detectAndCompute(MatVector images) {
        KeyPointVectorVector keypoints = new KeyPointVectorVector();
        feature2D.detect(images, keypoints);
        MatVector descriptors = new MatVector();
        feature2D.compute(images, keypoints, descriptors);
        return descriptors;
    }

    public Mat detectAndCompute(Mat image) {
        KeyPointVector keypoint = new KeyPointVector();
        feature2D.detect(image, keypoint);
        Mat descriptor = new Mat();
        feature2D.compute(image, keypoint, descriptor);
        return descriptor;
    }

    public String getDescriptorEnding() {
        return "." + detectorType.name().toLowerCase() + ".descr";
    }

    public DetectorType getDetectorType() {
        return detectorType;
    }

    public static enum DetectorType {
        SURF, ORB;
    }

}
