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

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Lars Oetermann <lars.oetermann.com>
 */
public class DescriptorExtractorWrapper {

    private final FeatureDetector featureDetector;
    private final DescriptorExtractor descriptorExtractor;

    public DescriptorExtractorWrapper(int detectorType, int extractorType) {
        featureDetector = FeatureDetector.create(detectorType);
        descriptorExtractor = DescriptorExtractor.create(extractorType);
    }
    
    public DescriptorExtractorWrapper() {
        this(FeatureDetector.ORB, FeatureDetector.ORB);
    }

    public List<Mat> readImages(List<String> files, boolean grayscale) {
        List<Mat> images = new ArrayList<>();
        Mat mat;

        for (ListIterator<String> it = files.listIterator(); it.hasNext();) {
            String file = it.next();
            mat = Imgcodecs.imread(file);
            if (mat.dims() > 0 && mat.cols() > 0 && mat.rows() > 0) {
                if (grayscale) {
                    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY);
                }
                images.add(mat);
            } else {
                it.remove();
                System.out.println("Cannot read file: " + file);
            }
        }
        return images;
    }

    public List<Mat> detectAndCompute(List<String> images, boolean grayscale) {
        List<Mat> imageList = readImages(images, grayscale);
        List<Mat> descriptors = detectAndCompute(imageList);
        imageList.stream().forEach((image) -> {
            image.release();
        });
        return descriptors;
    }

    public List<Mat> detectAndCompute(List<Mat> images) {
        List<MatOfKeyPoint> keypoints = new ArrayList<>();
        featureDetector.detect(images, keypoints);
        List<Mat> descriptors = new ArrayList<>();
        descriptorExtractor.compute(images, keypoints, descriptors);
        keypoints.stream().forEach((keypoint) -> {
            keypoint.release();
        });
        return descriptors;
    }

    public Mat detectAndCompute(Mat image) {
        MatOfKeyPoint keypoint = new MatOfKeyPoint();
        featureDetector.detect(image, keypoint);
        Mat descriptor = new Mat();
        descriptorExtractor.compute(image, keypoint, descriptor);
        keypoint.release();
        return descriptor;
    }

}
