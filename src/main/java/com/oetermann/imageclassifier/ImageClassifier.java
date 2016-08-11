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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_imgproc;

/**
 *
 * @author Lars Oetermann <lars.oetermann.com>
 */
public class ImageClassifier {

    public static final int NO_MATCH = -1, UNKOWN_MATCHER = -2;

    private final DescriptorExtractorWrapper descriptorExtractorWrapper;
    private final HashMap<String, MatchFinderWrapper> flannMatchers;

    public ImageClassifier(String extractorType) {
        this.descriptorExtractorWrapper = new DescriptorExtractorWrapper(extractorType);
        this.flannMatchers = new HashMap<>();
    }

    public ImageClassifier() {
        this.descriptorExtractorWrapper = new DescriptorExtractorWrapper();
        this.flannMatchers = new HashMap<>();
    }

    public void trainMatcher(String name, boolean recursivly, boolean grayscale, String... paths) {
        List<String> imgPaths = new ArrayList<>();
        for (String path : paths) {
            imgPaths.addAll(Util.listFiles(path, recursivly, ".jpg", ".jpeg", ".png", ".gif"));
        }
        trainMatcher(name, imgPaths, grayscale);
    }

    public void trainMatcher(String name, boolean recursivly, boolean grayscale, List<String> paths) {
        List<String> imgPaths = new ArrayList<>();
        paths.stream().forEach((path) -> {
            imgPaths.addAll(Util.listFiles(path, recursivly, ".jpg", ".jpeg", ".png", ".gif"));
        });
        trainMatcher(name, imgPaths, grayscale);
    }

    public void trainMatcher(String name, List<String> images, boolean grayscale) {
        trainMatcher(name, images, descriptorExtractorWrapper.detectAndCompute(images, grayscale));
    }

    public void trainMatcher(String name, List<String> files, MatVector descriptors) {
        if (flannMatchers.containsKey(name)) {
//            flannMatchers.get(name).release();
        }
        flannMatchers.put(name, new MatchFinderWrapper(files, descriptors));
//        descriptors.stream().forEach((descriptor) -> {
//            descriptor.release();
//        });
    }

    public void trainMatcherWithDescriptors(String name, boolean recursivly, String... descriptors) {
        List<String> descriptorFiles = new ArrayList<>();
        for (String descriptor : descriptors) {
            descriptorFiles.addAll(Util.listFiles(descriptor, recursivly, descriptorExtractorWrapper.getDescriptorEnding()));
        }
        trainMatcherWithDescriptors(name, descriptorFiles);
    }

    public void trainMatcherWithDescriptors(String name, boolean recursivly, List<String> descriptors) {
        List<String> descriptorFiles = new ArrayList<>();
        descriptors.stream().forEach((path) -> {
            descriptorFiles.addAll(Util.listFiles(path, recursivly, descriptorExtractorWrapper.getDescriptorEnding()));
        });
        trainMatcherWithDescriptors(name, descriptorFiles);
    }

    public void trainMatcherWithDescriptors(String name, List<String> descriptors) {
        MatVector descriptorList = new MatVector(descriptors.size());
        long i = 0;
        for (String descriptor : descriptors) {
            Mat descriptorMat = Util.loadMat(descriptor);
            if (descriptorMat != null) {
                descriptorList.put(i, descriptorMat);
                i++;
            }
        }
        descriptorList.resize(i);
        trainMatcher(name, descriptors, descriptorList);
    }

    public void precomputeDescriptors(boolean recursivly, String outputPath, boolean grayscale, String... inputPaths) {
        List<String> images = new ArrayList<>();

        for (String inputPath : inputPaths) {
            images.addAll(Util.listFiles(inputPath, recursivly, ".jpg", ".jpeg", ".png", ".gif"));
        }
        precomputeDescriptors(images, outputPath, grayscale);
    }

    public void precomputeDescriptors(boolean recursivly, String outputPath, boolean grayscale, List<String> inputPaths) {
        List<String> images = new ArrayList<>();
        inputPaths.stream().forEach((path) -> {
            images.addAll(Util.listFiles(path, recursivly, ".jpg", ".jpeg", ".png", ".gif"));
        });
        precomputeDescriptors(images, outputPath, grayscale);
    }

    public void precomputeDescriptors(List<String> images, String outputPath, boolean grayscale) {
        MatVector descriptors = descriptorExtractorWrapper.detectAndCompute(images, grayscale);
        String baseDir = Util.longestCommonPrefix(images);
        if (!outputPath.endsWith("/")) {
            outputPath += "/";
        }
        for (int i = 0; i < images.size(); i++) {
            Util.saveMat(outputPath + images.get(i).replaceFirst(baseDir, "")
                    + descriptorExtractorWrapper.getDescriptorEnding(), descriptors.get(i));
            descriptors.get(i).release();
        }
    }

    public int match(String matcherName, Mat queryImage) {
        return match(matcherName, queryImage, 22);
    }

    public int match(String matcherName, Mat queryImage, int minMatches) {
        if (!flannMatchers.containsKey(matcherName)) {
            return UNKOWN_MATCHER;
        }
        if (queryImage.type() == opencv_core.CV_8U) {
            opencv_imgproc.equalizeHist(queryImage, queryImage);
        }
//        long t = System.currentTimeMillis();
        Mat queryDescriptors = descriptorExtractorWrapper.detectAndCompute(queryImage);
//        System.out.println("SURF: "+(System.currentTimeMillis()-t));
//        t = System.currentTimeMillis();
        int match = flannMatchers.get(matcherName).bestMatch(queryDescriptors, minMatches);
        queryDescriptors.release();
//        System.out.println("FLANN: "+(System.currentTimeMillis()-t));
        return match;
    }

    public String matchName(String matcherName, Mat queryImage, int minMatches) {
        if (!flannMatchers.containsKey(matcherName)) {
            return "Unkown Matcher: " + matcherName;
        }
        return flannMatchers.get(matcherName).nameOf(match(matcherName, queryImage, minMatches));
    }

    public String matchName(String matcherName, byte[] jpegByte, int minMatches) {
        return matchName(matcherName, Util.fromByteJPEG(jpegByte, false), minMatches);
    }

    public HashMap<String, MatchFinderWrapper> getFlannMatchers() {
        return flannMatchers;
    }

    public DescriptorExtractorWrapper getDescriptorExtractorWrapper() {
        return descriptorExtractorWrapper;
    }

}
