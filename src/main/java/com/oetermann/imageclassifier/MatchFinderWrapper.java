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
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.features2d.DescriptorMatcher;

/**
 *
 * @author Lars Oetermann <lars.oetermann.com>
 */
public class MatchFinderWrapper {

    private final DescriptorMatcher matcher;
    private final String[] imageNames;
    private final double[] matchesPerImage;

    public MatchFinderWrapper(String fromFile) {
        matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        matcher.read(fromFile);
        this.matchesPerImage = new double[matcher.getTrainDescriptors().size()];
        imageNames = new String[matcher.getTrainDescriptors().size()];
        for (int i = 0; i < imageNames.length; i++) {
            imageNames[i] = "Image#" + i;
        }
    }

    public MatchFinderWrapper(List<String> images, List<Mat> descriptors) {
        matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        descriptors.stream().forEach((descriptor) -> {
            descriptor.convertTo(descriptor, CvType.CV_32F);
        });
        matcher.add(descriptors);
        matcher.train();
        matchesPerImage = new double[descriptors.size()];
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
        queryDescriptors.convertTo(queryDescriptors, CvType.CV_32F);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(queryDescriptors, matches);
        queryDescriptors.empty(); // Attempt to stop GC from releasing mat
        Arrays.fill(matchesPerImage, (short) 0);
        DMatch[] matchesArray = matches.toArray();
        for (DMatch match : matchesArray) {
//            match.distance;
            if (match.distance > 1) {
                match.distance = match.distance / 10000 - 1f;
            }
            if (match.distance < 0.5) {
                matchesPerImage[match.imgIdx] += 0.5 - match.distance;
            }
//            matchesPerImage[match.imgIdx] += 1;
//            System.out.println("ImgIndex: "+ match.imgIdx + " TrainIndex: " + match.trainIdx +" MatchDistance: "+match.distance);
        }
        int index = 0;
        for (int i = 0; i < matchesPerImage.length; i++) {
            System.out.println("Image #" + i + " has " + matchesPerImage[i] + " matches");
            if (matchesPerImage[i] > matchesPerImage[index]) {
                index = i;
            }
        }
        if (matchesPerImage[index] >= minMatches) {
            return index;
        }
        return -1;
    }

    public void release() {
        matcher.getTrainDescriptors().stream().forEach((trainDescriptor) -> {
            trainDescriptor.release();
        });
        matcher.clear();
    }

    public void save(String toFile) {
        matcher.write(toFile);
    }

}
