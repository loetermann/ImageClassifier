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
import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 *
 * @author Lars Oetermann <lars.oetermann.com>
 */
public class ImageClassifier {

    public static final int NO_MATCH = -1, UNKOWN_MATCHER = -2;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private final DescriptorExtractorWrapper surfDescriptorExtractor;
    private final HashMap<String, MatchFinderWrapper> flannMatchers;

    public ImageClassifier(int detectorType, int extractorType) {
        this.surfDescriptorExtractor = new DescriptorExtractorWrapper(detectorType, extractorType);
        this.flannMatchers = new HashMap<>();
    }
    
    public ImageClassifier() {
        this.surfDescriptorExtractor = new DescriptorExtractorWrapper();
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
        trainMatcher(name, images, surfDescriptorExtractor.detectAndCompute(images, grayscale));
    }

    public void trainMatcher(String name, List<String> files, List<Mat> descriptors) {
        if (flannMatchers.containsKey(name)) {
            flannMatchers.get(name).release();
        }
        flannMatchers.put(name, new MatchFinderWrapper(files, descriptors));
        descriptors.stream().forEach((descriptor) -> {
            descriptor.release();
        });
    }

    public void trainMatcherWithDescriptors(String name, boolean recursivly, String... descriptors) {
        List<String> descriptorFiles = new ArrayList<>();
        for (String descriptor : descriptors) {
            descriptorFiles.addAll(Util.listFiles(descriptor, recursivly, ".descr"));
        }
        trainMatcherWithDescriptors(name, descriptorFiles);
    }

    public void trainMatcherWithDescriptors(String name, boolean recursivly, List<String> descriptors) {
        List<String> descriptorFiles = new ArrayList<>();
        descriptors.stream().forEach((path) -> {
            descriptorFiles.addAll(Util.listFiles(path, recursivly, ".descr"));
        });
        trainMatcherWithDescriptors(name, descriptorFiles);
    }

    public void trainMatcherWithDescriptors(String name, List<String> descriptors) {
        List<Mat> descriptorList = new ArrayList<>();
        descriptors.stream().forEach((descriptor) -> {
            Mat descriptorMat = Util.loadMat(descriptor);
            if (descriptorMat != null) {
                descriptorList.add(descriptorMat);
            }
        });
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
        List<Mat> descriptors = surfDescriptorExtractor.detectAndCompute(images, grayscale);
        String baseDir = Util.longestCommonPrefix(images);
        if (!outputPath.endsWith("/")) {
            outputPath += "/";
        }
        for (int i = 0; i < images.size(); i++) {
            Util.saveMat(outputPath + images.get(i).replaceFirst(baseDir, "") + ".descr", descriptors.get(i));
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
        long t = System.currentTimeMillis();
        Mat queryDescriptors = surfDescriptorExtractor.detectAndCompute(queryImage);
        System.out.println("SURF: "+(System.currentTimeMillis()-t));
        t = System.currentTimeMillis();
        int match = flannMatchers.get(matcherName).bestMatch(queryDescriptors, minMatches);
        queryDescriptors.release();
        System.out.println("FLANN: "+(System.currentTimeMillis()-t));
        return match;
    }

    public String matchName(String matcherName, Mat queryImage, int minMatches) {
        if (!flannMatchers.containsKey(matcherName)) {
            return "Unkown Matcher: " + matcherName;
        }
        return flannMatchers.get(matcherName).nameOf(match(matcherName, queryImage, minMatches));
    }

}
