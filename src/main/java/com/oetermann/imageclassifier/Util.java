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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 *
 * @author Lars Oetermann <lars.oetermann.com>
 */
public interface Util {

    public static String longestCommonPrefix(List<String> strings) {
        if (strings.isEmpty()) {
            return "";   // Or maybe return null?
        }

        for (int prefixLen = 0; prefixLen < strings.get(0).length(); prefixLen++) {
            char c = strings.get(0).charAt(prefixLen);
            for (String string : strings) {
                if (prefixLen >= string.length()
                        || string.charAt(prefixLen) != c) {
                    // Mismatch found
                    return string.substring(0, prefixLen);
                }
            }
        }
        return strings.get(0);
    }

    public static List<String> listFiles(String path, boolean recursive, String... acceptedEndings) {
        List<String> files = new ArrayList<>();
        File file = new File(path);
        if (file.isDirectory()) {
            LinkedList<File> directories = new LinkedList<>();
            directories.add(file);
            while (!directories.isEmpty()) {
                file = directories.poll();
                for (File child : file.listFiles()) {
                    if (child.isDirectory() && recursive) {
                        directories.add(child);
                    } else if (child.isFile()
                            && (Arrays.stream(acceptedEndings).anyMatch(child.getName()::endsWith)
                            || acceptedEndings.length == 0)) {
                        files.add(child.getPath());
                    }
                }
            }
        } else {
            files.add(path);
        }
        return files;
    }

    public static void saveMat(String path, Mat mat) {
        File file = new File(path).getAbsoluteFile();
        file.getParentFile().mkdirs();
        try {
            int rows = mat.rows();
            int cols = mat.cols();
            int type = mat.type();
            Object data;
            switch (mat.type()) {
                case CvType.CV_8S:
                case CvType.CV_8U:
                    data = new byte[(int) mat.total() * mat.channels()];
                    mat.get(0, 0, (byte[]) data);
                    break;
                case CvType.CV_16S:
                case CvType.CV_16U:
                    data = new short[(int) mat.total() * mat.channels()];
                    mat.get(0, 0, (short[]) data);
                    break;
                case CvType.CV_32S:
                    data = new int[(int) mat.total() * mat.channels()];
                    mat.get(0, 0, (int[]) data);
                    break;
                case CvType.CV_32F:
                    data = new float[(int) mat.total() * mat.channels()];
                    mat.get(0, 0, (float[]) data);
                    break;
                case CvType.CV_64F:
                    data = new double[(int) mat.total() * mat.channels()];
                    mat.get(0, 0, (double[]) data);
                    break;
                default:
                    data = null;
            }
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
                oos.writeObject(rows);
                oos.writeObject(cols);
                oos.writeObject(type);
                oos.writeObject(data);
                oos.close();
            }
        } catch (IOException | ClassCastException ex) {
            System.err.println("ERROR: Could not save mat to file: " + path);
//            Logger.getLogger(ImageClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static Mat loadMat(String path) {
        try {
            int rows, cols, type;
            Object data;
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
                rows = (int) ois.readObject();
                cols = (int) ois.readObject();
                type = (int) ois.readObject();
                data = ois.readObject();
            }
            Mat mat = new Mat(rows, cols, type);
            switch (type) {
                case CvType.CV_8S:
                case CvType.CV_8U:
                    mat.put(0, 0, (byte[]) data);
                    break;
                case CvType.CV_16S:
                case CvType.CV_16U:
                    mat.put(0, 0, (short[]) data);
                    break;
                case CvType.CV_32S:
                    mat.put(0, 0, (int[]) data);
                    break;
                case CvType.CV_32F:
                    mat.put(0, 0, (float[]) data);
                    break;
                case CvType.CV_64F:
                    mat.put(0, 0, (double[]) data);
                    break;
            }
            return mat;
        } catch (IOException | ClassNotFoundException | ClassCastException ex) {
            System.err.println("ERROR: Could not load mat from file: " + path);
//            Logger.getLogger(ImageClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
}
