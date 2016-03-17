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

    private final SurfDescriptorExtractor surfDescriptorExtractor;
    private final HashMap<String, FlannMatchFinder> flannMatchers;

    public ImageClassifier() {
        this.surfDescriptorExtractor = new SurfDescriptorExtractor();
        this.flannMatchers = new HashMap<>();
    }

    public void trainMatcher(String name, String path, boolean recursivly, boolean grayscale) {
        trainMatcher(name, Util.listFiles(path, recursivly), grayscale);
    }

    public void trainMatcher(String name, List<String> images, boolean grayscale) {
        trainMatcher(name, images, surfDescriptorExtractor.detectAndCompute(images, grayscale));
    }

    public void trainMatcher(String name, List<String> files, List<Mat> descriptors) {
        if (flannMatchers.containsKey(name)) {
            flannMatchers.get(name).release();
        }
        flannMatchers.put(name, new FlannMatchFinder(files, descriptors));
        descriptors.stream().forEach((descriptor) -> {
            descriptor.release();
        });
    }

    public void trainMatcherWithDescriptors(String name, List<String> descriptors) {
        List<Mat> descriptorList = new ArrayList<>();
        descriptors.stream().forEach((descriptor) -> {
            descriptorList.add(Util.loadMat(descriptor));
        });
        trainMatcher(name, descriptors, descriptorList);
    }

    public void precomputeDescriptors(String inputPath, boolean recursivly, String outputPath, boolean grayscale) {
        precomputeDescriptors(Util.listFiles(inputPath, recursivly), outputPath, grayscale);
    }

    public void precomputeDescriptors(List<String> images, String outputPath, boolean grayscale) {
        List<Mat> descriptors = surfDescriptorExtractor.detectAndCompute(images, grayscale);
        String baseDir = Util.longestCommonPrefix(images);
        for (int i = 0; i < images.size(); i++) {
            Util.saveMat(images.get(i).replaceFirst(baseDir, ""), descriptors.get(i));
            descriptors.get(i).release();
        }
    }
    
    public int match(String matcherName, Mat queryImage) {
        return match(matcherName, queryImage, 22);
    }
    
    public int match(String matcherName, Mat queryImage, int minMatches) {
        if(!flannMatchers.containsKey(matcherName)) {
            return UNKOWN_MATCHER;
        }
        Mat queryDescriptors = surfDescriptorExtractor.detectAndCompute(queryImage);
        int match = flannMatchers.get(matcherName).bestMatch(queryDescriptors, minMatches);
        queryDescriptors.release();
        return match;
    }
        
    public String matchName(String matcherName, Mat queryImage, int minMatches) {
        if(!flannMatchers.containsKey(matcherName)) {
            return "Unkown Matcher: "+matcherName;
        }
        return flannMatchers.get(matcherName).nameOf(match(matcherName, queryImage, minMatches));
    }

}
