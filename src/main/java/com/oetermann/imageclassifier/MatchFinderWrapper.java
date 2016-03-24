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

import java.util.Arrays;
import java.util.List;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.DMatch;
import org.bytedeco.javacpp.opencv_core.DMatchVector;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_features2d.DescriptorMatcher;
/**
 *
 * @author Lars Oetermann <lars.oetermann.com>
 */
public class MatchFinderWrapper {

    private final DescriptorMatcher matcher;
    private final String[] imageNames;
    private final double[] matchesPerImage;

//    public MatchFinderWrapper(String fromFile) {
//        matcher = new opencv_features2d.FlannBasedMatcher();
////        matcher.read(fromFile);
//        this.matchesPerImage = new double[matcher.getTrainDescriptors().size()];
//        imageNames = new String[matcher.getTrainDescriptors().size()];
//        for (int i = 0; i < imageNames.length; i++) {
//            imageNames[i] = "Image#" + i;
//        }
//    }

    public MatchFinderWrapper(List<String> images, MatVector descriptors) {
        matcher = new opencv_features2d.FlannBasedMatcher();
        for (int i = 0; i < descriptors.size(); i++) {
            descriptors.get(i).convertTo(descriptors.get(i), opencv_core.CV_32F);
        }
        matcher.add(descriptors);
        matcher.train();
        matchesPerImage = new double[(int)descriptors.size()];
        imageNames = new String[images.size()];
        for (int i = 0; i < images.size(); i++) {
            String name = images.get(i);
            imageNames[i] = name.substring(name.lastIndexOf('/') + 1, name.contains(".descr") ? name.indexOf(".descr") : name.length());
        }
    }

    public String nameOf(int i) {
        if (i < 0) {
            return "No match found.";
        }
        return imageNames[i];
    }

    public int bestMatch(Mat queryDescriptors, int minMatches) {
        queryDescriptors.convertTo(queryDescriptors, opencv_core.CV_32F);
        DMatchVector matches = new DMatchVector();
        matcher.match(queryDescriptors, matches);
        queryDescriptors.empty(); // Attempt to stop GC from releasing mat
        Arrays.fill(matchesPerImage, 0);
        for (int i = 0; i < matches.size(); i++) {
            DMatch match = matches.get(i);
//            match.distance;
            float distance = match.distance();
            if (distance > 1) {
                distance = distance / 1000;
            }
            if (distance < 1) {
                matchesPerImage[match.imgIdx()] += 1 - distance;
            }
//            matchesPerImage[match.imgIdx] += 1;
//            System.out.println("MatchDistance: "+match.distance + "\t\tImage: "+ imageNames[match.imgIdx]);
        }
        int index = 0;
        for (int i = 0; i < matchesPerImage.length; i++) {
//            System.out.println(matchesPerImage[i] + "\t\tmatches for image " + imageNames[i]);
            if (matchesPerImage[i] > matchesPerImage[index]) {
                index = i;
            }
        }
//        System.out.println("Total Matches: "+matches.size());
        if (matchesPerImage[index] >= minMatches) {
            return index;
        }
        return -1;
    }

//    public void release() {
//        matcher.getTrainDescriptors().stream().forEach((trainDescriptor) -> {
//            trainDescriptor.release();
//        });
//        matcher.clear();
//    }

//    public void save(String toFile) {
//        matcher.write(toFile);
//    }

}
