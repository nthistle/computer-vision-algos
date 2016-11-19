import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.util.Random;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Scanner;
import java.lang.Math;


public class ComputerVision
{

   public static final int BIN_SIZE = 256; // decimal grayscaled means this could be higher
   
   public static void main(String[] args) {
      if(args.length < 2) {
         System.out.println("usage: java ComputerVision {edge|bin} image_name [gaussian_level]");
         System.exit(-1);
      }
      String command = args[0];
      if(command.equalsIgnoreCase("edgedetect") || command.equalsIgnoreCase("edge")) {
         doEdgeDetect(args);
      }
      else if(command.equalsIgnoreCase("binarize") || command.equalsIgnoreCase("bin")) {
         doBinarize(args);
      }
      else if(command.equalsIgnoreCase("dog")) {
         differenceOfGaussians(args);
      }
   }
   
   public static void differenceOfGaussians(String[] args) {
      int NUM_GAUSS = (args.length > 2)?Integer.parseInt(args[2]):3;
   
      int pos = args[1].lastIndexOf(".");
      String base_name = args[1].substring(0,pos);
   
      System.out.println("Reading image...");
      BufferedImage testImage = readImage(args[1]);
      System.out.println("Grayscaling...");
      double[][] gray = grayscale(testImage);
      
      System.out.println("Converting to image and writing...");
      BufferedImage grayImage = grayToImage(gray);
      writeImage(grayImage,base_name + "_gray.png");
   
      double[][] current = cloneArr(gray);
      if(NUM_GAUSS>0) {
         System.out.println("Applying Gaussian Blur...");
         for(int i = 0; i < NUM_GAUSS; i ++) {
            current = gaussian(current);
         }
      }
      System.out.println("Subtracting...");
      double[][] dog = difference(current, gray);
      double[][] normalizedDog = normalize(dog);
            
      System.out.println("Converting to image and writing...");
      BufferedImage dogImage = grayToImage(normalizedDog);
      writeImage(dogImage, base_name + "_difference_of_gauss.png");
   }
   
   public static void doBinarize(String[] args) {
   
      int pos = args[1].lastIndexOf(".");
      String base_name = args[1].substring(0,pos);
   
      System.out.println("Reading image...");
      BufferedImage testImage = readImage(args[1]);
      System.out.println("Grayscaling...");
      double[][] gray = grayscale(testImage);
      
      System.out.println("Converting to image and writing...");
      BufferedImage grayImage = grayToImage(gray);
      writeImage(grayImage,base_name + "_gray.png");
   
      System.out.println("Calculating Histogram...");
      int[] hist = getHistogram(gray, BIN_SIZE);
         
      System.out.println("Using Otsu Thresholding...");
      int threshold = otsuThreshold(hist);
         
      System.out.println("Drawing histogram...");
       
      BufferedImage histImage = histToImage(hist, threshold);
      writeImage(histImage, base_name + "_gray_histogram.png");
         
      System.out.println("Found Threshold Value of " + threshold); // will already be in [0,255] range
      System.out.println("Binarizing Image with Determined Threshold...");
      double[][] binary = binarize(gray, ((double)threshold)/BIN_SIZE);
      System.out.println("Converting to image and writing...");
      BufferedImage binaryImage = grayToImage(binary);
      writeImage(binaryImage, base_name + "_binary.png");
      
   }
   
   public static void doEdgeDetect(String[] args) {
      int NUM_GAUSS = (args.length > 2)?Integer.parseInt(args[2]):0;
      
      int pos = args[1].lastIndexOf(".");
      String base_name = args[1].substring(0,pos);
   
      System.out.println("Reading image...");
      BufferedImage testImage = readImage(args[1]);
      System.out.println("Grayscaling...");
      double[][] gray = grayscale(testImage);
      
      System.out.println("Converting to image and writing...");
      BufferedImage grayImage = grayToImage(gray);
      writeImage(grayImage,base_name + "_gray.png");

      double[][] current = gray;
      if(NUM_GAUSS>0) {
         System.out.println("Applying Gaussian Blur...");
         for(int i = 0; i < NUM_GAUSS; i ++) {
            current = gaussian(current);
         }
      }
      
      System.out.println("Applying Sobel...");
      double[][][] rawSobelIntensity = sobelRaw(current);
      System.out.println("Applying Non-maximum suppression...");
      double[][] thinned = nonmaxsuppression(rawSobelIntensity);
      double[][] normalThin = normalize(thinned);
      System.out.println("Converting to image and writing...");
      BufferedImage sobelImage = grayToImage(normalize(sobelRawToParsed(rawSobelIntensity)));
      writeImage(sobelImage, base_name + "_sobel.png");
      BufferedImage thinImage = grayToImage(normalThin);
      writeImage(thinImage, base_name + "_sobel_thin.png");
      
      System.out.println("Thresholding edges...");
      int[][] thresholded = thresholdedges(normalThin, 0.4, 0.7); // requires tweaking
      System.out.println("Applying Blob Analysis...");
      boolean[][] blobbed = blobAnalysis(thresholded);
      System.out.println("Converting to image and writing...");
      BufferedImage edgeImage = booleanToImage(blobbed);
      writeImage(edgeImage, base_name + "_finaledges.png");
      
      //int[] histEdges = getHistogram(normalThin);
      //int chosenThresh = otsuThreshold(histEdges);
      //writeImage(histToImage(histEdges, chosenThresh), base_name + "_edgeshist.png");
      //for(int i : histEdges) System.out.println(i);
      //System.out.println("Thresh: " + chosenThresh);
   }
   
  /**
   * Converts a histogram in the format of an array of frequencies 
   * into an image for easier user analysis, and draws a breakpoint
   * line, useful for analyzing accuracy of threshold choices.
   *
   * @param  hist        histogram stored as integer array of frequencies
   * @param  breakpoint  which bin of the histogram to draw the break line at
   * @return             BufferedImage with visual representation of histogram
   */
   public static BufferedImage histToImage(int[] hist, int breakpoint) {
      BufferedImage histImg = new BufferedImage(hist.length, hist.length+15, BufferedImage.TYPE_INT_ARGB);
      // scale to max of hist.length
      double[] scaledHist = new double[hist.length];
      int maxFreq = 0;
      for(int i = 0; i < hist.length; i ++) {
         if(hist[i] > maxFreq) maxFreq = hist[i];
      }
      for(int i = 0; i < hist.length; i ++) {
         scaledHist[i] = hist.length * hist[i] / ((double)maxFreq);
      }
      int gval;
      for(int i = 0; i < hist.length; i ++) {
         for(int j = 0; j < hist.length+15; j ++) {
            if(j >= hist.length) {
               gval = (int)((255.0*i)/(double)hist.length);
               setColor(histImg, i, j, new Color(gval, gval, gval));
            }
            else if(i == breakpoint) {
               setColor(histImg, i, j, new Color(0, 0, 0));
            }
            else if((hist.length-j) > scaledHist[i]) {
               setColor(histImg, i, j, new Color(255, 255, 255));
            }
            else {
               setColor(histImg, i, j, new Color(255, 0, 0));
            }
         }
      }
      return histImg;
   }

  /**
   * Converts a histogram in the format of an array of frequencies 
   * into an image for easier user analysis.
   *
   * @param  hist        histogram stored as integer array of frequencies
   * @return             BufferedImage with visual representation of histogram
   */
   public static BufferedImage histToImage(int[] hist) {
      BufferedImage histImg = new BufferedImage(hist.length, hist.length+15, BufferedImage.TYPE_INT_ARGB);
      // scale to max of hist.length
      double[] scaledHist = new double[hist.length];
      int maxFreq = 0;
      for(int i = 0; i < hist.length; i ++) {
         if(hist[i] > maxFreq) maxFreq = hist[i];
      }
      for(int i = 0; i < hist.length; i ++) {
         scaledHist[i] = hist.length * hist[i] / ((double)maxFreq);
      }
      int gval;
      for(int i = 0; i < hist.length; i ++) {
         for(int j = 0; j < hist.length+15; j ++) {
            if(j >= hist.length) {
               gval = (int)((255.0*i)/(double)hist.length);
               setColor(histImg, i, j, new Color(gval, gval, gval));
            }
            else if((hist.length-j) > scaledHist[i]) {
               setColor(histImg, i, j, new Color(255, 255, 255));
            }
            else {
               setColor(histImg, i, j, new Color(255, 0, 0));
            }
         }
      }
      return histImg;
   }
   
  /**
   * Inverts an image represented as a double 2D array, assuming
   * values in range [0,1] in input.
   *
   * @param  img         2D double array of values representing image, in range [0,1]
   * @return             inverted representation of image in 2D double array
   */
   public static double[][] invert(double[][] img) {
      double[][] inv = new double[img.length][img[0].length];
      for(int x = 0; x < img.length; x ++) {
         for(int y = 0; y < img[0].length; y ++) {
            inv[x][y] = 1.0 - img[x][y];
         }
      }
      return inv;
   }
   
  /**
   * Turns a "raw" matrix of Sobel directional gradients (Gx and Gy) into
   * a cleaner representation of the total gradient, sqrt(Gx^2+Gy^2), which
   * can directly be turned into a grayscale image.
   *
   * @param  rawSobel    3D double array of values representing sobel gradients (last index being Gx, Gy)
   * @return             cleaner representation of the total gradient
   */
   public static double[][] sobelRawToParsed(double[][][] rawSobel) {
      double[][] gvals = new double[rawSobel.length][rawSobel[0].length];
      for(int i = 0; i < rawSobel.length; i ++) {
         for(int j = 0; j < rawSobel[0].length; j ++) {
            gvals[i][j] = Math.sqrt(rawSobel[i][j][0]*rawSobel[i][j][0]+rawSobel[i][j][1]*rawSobel[i][j][1]);
         }
      }
      return gvals;
   }
   
  /**
   * Normalizes all the values in a given double array into the range [0,1],
   * which is then more readily turned into an image with the other methods.
   *
   * @param  raw         2D double array of values in any range
   * @return             normalized 2D double array to range [0,1]
   */
   public static double[][] normalize(double[][] raw) {
      double minval = Double.POSITIVE_INFINITY;
      double maxval = Double.NEGATIVE_INFINITY;
      double[][] normalized = new double[raw.length][raw[0].length];
      for(int i = 0; i < raw.length; i ++) {
         for(int j = 0; j < raw[0].length; j ++) {
            if(raw[i][j] < minval) minval = raw[i][j];
            if(raw[i][j] > maxval) maxval = raw[i][j];
         }
      }
      for(int i = 0; i < raw.length; i ++) {
         for(int j = 0; j < raw[0].length; j ++) {
            normalized[i][j] = (raw[i][j]-minval)/(maxval-minval);
         }
      }
      return normalized;
   }
   
   public static double[][] difference(double[][] m1, double[][] m2) {
      double[][] diff = new double[m1.length][m1[0].length];
      for(int i = 0; i < m1.length; i ++) {
         for(int j = 0; j < m1[0].length; j ++) {
            diff[i][j] = m1[i][j] - m2[i][j];
         }      
      }
      return diff;
   }
   
   public static double[][] cloneArr(double[][] m1) {
      double[][] clone = new double[m1.length][m1[0].length];
      for(int i = 0; i < m1.length; i ++) {
         for(int j = 0; j < m1[0].length; j ++) {
            clone[i][j] = m1[i][j];
         }      
      }
      return clone;
   }
   
  /**
   * Returns a histogram of the supplied grayscaled image data, into a default
   * of 256 bins
   *
   * @param  img         2D double array of values to be used in histogram
   * @return             int array of histogram representing frequencies of value ranges
   */
   public static int[] getHistogram(double[][] img) {
      return getHistogram(img, 256);
   }
   
   
  /**
   * Returns a histogram of the supplied grayscaled image data, into the number
   * of supplied bins
   *
   * @param  img         2D double array of values to be used in histogram
   * @param  bins        number of "bins" to group data into
   * @return             int array of length bins of histogram representing frequencies of value ranges
   */
   public static int[] getHistogram(double[][] img, int bins) {
      int[] binned = new int[bins];
      // assuming img is in range from [0,1]
      for(double[] row : img) {
         for(double d : row) {
            binned[(int)((bins-1)*d)] ++;
         }
      }
      return binned;
   }
   
  /**
   * Binarizes a supplied double array (image representation) according to supplied
   * threshold, values below threshold become 0, values above become 1
   *
   * @param  grayscale   2D double array representing grayscaled image (preferably in range [0,1])
   * @param  threshold   threshold on which to binarize supplied data
   * @return             binarized 2D double array with values 0 and 1
   */
   public static double[][] binarize(double[][] grayscale, double threshold) {
      double[][] binaryimg = new double[grayscale.length][grayscale[0].length];
      for(int i = 0; i < grayscale.length; i ++) {
         for(int j = 0; j < grayscale[0].length; j ++) {
            if(grayscale[i][j] > threshold)
               binaryimg[i][j] = 1.0;
            else
               binaryimg[i][j] = 0.0;
         }
      }
      return binaryimg; 
   }
   
   
  /**
   * Analyzes a histogram and returns a threshold that maximizes between-class variance
   * (also minimizing within class variance)
   * heavily based off of http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html    
   * 
   * @param  histogram   integer array representing frequencies in corresponding bins
   * @return             ideal threshold value (bin #) to divide data to maximize between-class variance
   */
   public static int otsuThreshold(int[] histogram) {
      double sum = 0.0;
      int total = 0;
      for(int i = 0; i < histogram.length; i ++) {
         sum += i*histogram[i];
         total += histogram[i];
      }
      // A region is the lower side (black),
      // B region is the upper side (white)
      
      double sumA = 0.0;
      int weightA = 0;
      int weightB = 0;
      
      double maxVariance = 0;
      int bestThreshold = 0;
      
      double meanA, meanB, variance;
      
      for(int i = 0; i < histogram.length; i ++) {
         weightA += histogram[i];
         if(weightA == 0) continue;
         
         weightB = total - weightA;
         if(weightB == 0) break;
         
         sumA += i*histogram[i];
         // sumB is sum-sumA
         
         meanA = sumA / weightA;
         meanB = (sum - sumA) / weightB;
         
         variance = (double)weightA * (double)weightB * (meanA - meanB) * (meanA - meanB);
         
         if(variance > maxVariance) {
            maxVariance = variance;
            bestThreshold = i;
         }
      }
      
      return bestThreshold;
   }
   
   
  /**
   * Turns a representation of "strong" and "weak" edges into a regular representation
   * of edges by only keeping strong edges and weak edges adjacent to strong edges
   *
   * @param  thresholded 2D integer array representation of "strong" and "weak" edges
   * @return             "blob analyzed" boolean area with determined edges
   */
   public static boolean[][] blobAnalysis(int[][] thresholded) {
      // keeps weak edges only if adjacent to strong edge
      boolean[][] edges = new boolean[thresholded.length][thresholded[0].length];
      for(int i = 0; i < thresholded.length; i ++) {
         for(int j = 0; j < thresholded[0].length; j ++) {
            edges[i][j] = false;
            if(thresholded[i][j] == 2) {
               edges[i][j] = true;
            } else if(thresholded[i][j] == 1) {
               if(i > 0 && thresholded[i-1][j] == 2) edges[i][j] = true;
               else if(j > 0 && thresholded[i][j-1] == 2) edges[i][j] = true;
               else if(i < thresholded.length-1 && thresholded[i+1][j] == 2) edges[i][j] = true;
               else if(j < thresholded[0].length-1 && thresholded[i][j+1] == 2) edges[i][j] = true;
            }
         }
      }
      return edges;
   }
   
   public static BufferedImage booleanToImage(boolean[][] thresholded) {
      BufferedImage img = new BufferedImage(thresholded.length, thresholded[0].length, BufferedImage.TYPE_INT_ARGB);
      for(int x = 0; x < thresholded.length; x ++) {
         for(int y = 0; y < thresholded[0].length; y ++) {
            if(thresholded[x][y])
               setColor(img, x, y, new Color(255,255,255));
            else
               setColor(img, x, y, new Color(0,0,0));
         }
      }
      return img;
   }
   
   public static int[][] thresholdedges(double[][] edges, double strongThresh, double weakThresh) {
      // 0 = none
      // 1 = weak
      // 2 = strong
      int[][] thresholded = new int[edges.length][edges[0].length];
      for(int i = 0; i < edges.length; i ++) {
         for(int j = 0; j < edges[0].length; j ++) {
            if(edges[i][j] > strongThresh)
               thresholded[i][j] = 2;
            else if(edges[i][j] > weakThresh)
               thresholded[i][j] = 1;
            else
               thresholded[i][j] = 0;
         }
      }
      return thresholded;
   }
   
   public static double[][] nonmaxsuppression(double[][][] rawSobel) {
      double[][] thinned = new double[rawSobel.length][rawSobel[0].length];
      double[][] gvals = new double[rawSobel.length][rawSobel[0].length];
      for(int i = 0; i < rawSobel.length; i ++) {
         for(int j = 0; j < rawSobel[0].length; j ++) {
            gvals[i][j] = Math.sqrt(rawSobel[i][j][0]*rawSobel[i][j][0]+rawSobel[i][j][1]*rawSobel[i][j][1]);
            thinned[i][j] = 0;
         }
      }
      //Random r = new Random();
      //double MYPROB = 0;
      
      double dir,myval;
      for(int i = 0; i < rawSobel.length; i ++) {
         for(int j = 0; j < rawSobel[0].length; j ++) {
            dir = (Math.PI + Math.atan2(rawSobel[i][j][0], rawSobel[i][j][1])) % Math.PI;
            myval = gvals[i][j];
            //if(myval > 0.01) {
            //   System.out.println("dat:");
            //   System.out.println(dir);
            //   System.out.println(myval);
            //}
            /*
              3   2   1 
               \  |  /
                \ | /  
                 \|/    
              4---X---0 
              
              now mapped to [0,+pi]
             */
            if(dir<(Math.PI/8.0) || dir>(7*Math.PI/8.0)) { // direction 0,4
               if(j>0&&j<rawSobel[0].length-1) {
                  //System.out.println("dir0 me: " + myval);
                  //System.out.println("neighboring vals: " + gvals[i][j-1] + "," + gvals[i][j+1]);
                  if(myval > gvals[i][j-1] && myval > gvals[i][j+1]) {
                     thinned[i][j] = myval;
                  }
               }
            }
            else if(dir<(3*Math.PI/8.0)) { // direction 1
               if(i>0&&i<rawSobel.length-1&&j>0&&j<rawSobel[0].length-1) {
                  if(myval > gvals[i-1][j-1] && myval > gvals[i+1][j+1]) {
                     thinned[i][j] = myval;
                  }
               }
            }
            else if(dir<(5*Math.PI/8.0)) { // direction 2
               if(i>0&&i<rawSobel.length-1) {
                  //System.out.println("dir2 me: " + myval);
                  //System.out.println("neighboring vals: " + gvals[i-1][j] + "," + gvals[i+1][j]);
                  if(myval > gvals[i-1][j] && myval > gvals[i+1][j]) {
                     thinned[i][j] = myval;
                  }
               }
            }
            else { // 7pi/8,  direction 3
               if(i>0&&i<rawSobel.length-1&&j>0&&j<rawSobel[0].length-1) {
                  if(myval > gvals[i-1][j+1] && myval > gvals[i+1][j-1]) {
                     thinned[i][j] = myval;
                  }
               }
            }
         }
      }
      return thinned;
   }
   
   public static double[][] gaussian(double[][] normal) {
      double[][] gauss = new double[normal.length][normal[0].length];
      double gaussval;
      for(int i = 0; i < normal.length; i ++) {
         for(int j = 0; j < normal[0].length; j ++) {
            if(i == 0 || j == 0 || i == normal.length-1 || j == normal[0].length-1) {
               gauss[i][j] = normal[i][j]; // too lazy to do a half blur
               continue;
            }
            gaussval = 0.0;
            gaussval += normal[i-1][j-1] + normal[i+1][j-1] + normal[i-1][j+1] + normal[i+1][j+1];
            gaussval += 2 * (normal[i-1][j] + normal[i+1][j] + normal[i][j-1] + normal[i][j+1]);
            gaussval += 4 * normal[i][j];
            gauss[i][j] = gaussval / 16.0;
         }
      }
      return gauss;
   }
   
   
   public static BufferedImage grayToImage(double[][] grayscaled) {
      BufferedImage img = new BufferedImage(grayscaled.length, grayscaled[0].length, BufferedImage.TYPE_INT_ARGB);
      for(int x = 0; x < grayscaled.length; x ++) {
         for(int y = 0; y < grayscaled[0].length; y ++) {
            int value = (int)(255*grayscaled[x][y]);
            setColor(img, x, y, new Color(value, value, value));
         }
      }
      return img;
   }


   // note: normalizes to range [0,1] from [0,255], no additional scaling
   public static double[][] grayscale(BufferedImage img) {
      double[][] grayscaled = new double[img.getWidth()][img.getHeight()];
      for(int x = 0; x < img.getWidth(); x ++) {
         for(int y = 0; y < img.getHeight(); y ++) {
            Color pix = getColor(img, x, y);
            grayscaled[x][y] = (0.30 * pix.getRed() + 0.59 * pix.getGreen() + 0.11 * pix.getBlue())/255.0;
         }
      }
      return grayscaled;
   }


   public static double[][] sobelGradient(double[][] grayscaled) {
      double maxG = Double.NEGATIVE_INFINITY;
      double minG = Double.POSITIVE_INFINITY;
      double gx, gy;
      double[][] grad = new double[grayscaled.length][grayscaled[0].length];
      for(int i = 0; i < grayscaled.length; i ++) {
         for(int j = 0; j < grayscaled[0].length; j ++) {
            if(i == 0 || j == 0 || i == grayscaled.length-1 || j == grayscaled[0].length-1) {
               grad[i][j] = 0.0;
               continue;
            }  
            // uses the following matrices:
            // Gx = [[-1, 0, +1],
            //       [-2, 0, +2],
            //       [-1, 0, +1]]
            //
            // Gy = [[-1,-2,-1],
            //       [ 0, 0, 0],
            //       [+1,+2,+1]]
            gx = 0;
            gx += (-1 * grayscaled[i-1][j-1]);
            gx += (-2 * grayscaled[i-1][j]);
            gx += (-1 * grayscaled[i-1][j+1]);
            gx += (1 * grayscaled[i+1][j-1]);
            gx += (2 * grayscaled[i+1][j]);
            gx += (1 * grayscaled[i+1][j+1]);
            
            gy = 0;
            gy += (-1 * grayscaled[i-1][j-1]);
            gy += (-2 * grayscaled[i][j-1]);
            gy += (-1 * grayscaled[i+1][j-1]);
            gy += (1 * grayscaled[i-1][j+1]);
            gy += (2 * grayscaled[i][j+1]);
            gy += (1 * grayscaled[i+1][j+1]);
            
            grad[i][j] = Math.sqrt(gx*gx+gy*gy);
            if(grad[i][j] > maxG) maxG = grad[i][j];
            if(grad[i][j] < minG) minG = grad[i][j];
         }
      }
      for(int i = 0; i < grad.length; i ++) {
         for(int j = 0; j < grad[0].length; j ++) {
            grad[i][j] = (grad[i][j]-minG)/(maxG-minG); // again normalizing to [0,1] and scaling
         }
      }
      return grad;
   }




   public static double[][][] sobelRaw(double[][] grayscaled) {
      double gx, gy;
      double[][][] grad = new double[grayscaled.length][grayscaled[0].length][2];
      for(int i = 0; i < grayscaled.length; i ++) {
         for(int j = 0; j < grayscaled[0].length; j ++) {
            if(i == 0 || j == 0 || i == grayscaled.length-1 || j == grayscaled[0].length-1) {
               grad[i][j][0] = 0.0;
               grad[i][j][1] = 0.0;
               continue;
            }  
            // uses the following matrices:
            // Gx = [[-1, 0, +1],
            //       [-2, 0, +2],
            //       [-1, 0, +1]]
            //
            // Gy = [[-1,-2,-1],
            //       [ 0, 0, 0],
            //       [+1,+2,+1]]
            gx = 0;
            gx += (-1 * grayscaled[i-1][j-1]);
            gx += (-2 * grayscaled[i-1][j]);
            gx += (-1 * grayscaled[i-1][j+1]);
            gx += (1 * grayscaled[i+1][j-1]);
            gx += (2 * grayscaled[i+1][j]);
            gx += (1 * grayscaled[i+1][j+1]);
            
            gy = 0;
            gy += (-1 * grayscaled[i-1][j-1]);
            gy += (-2 * grayscaled[i][j-1]);
            gy += (-1 * grayscaled[i+1][j-1]);
            gy += (1 * grayscaled[i-1][j+1]);
            gy += (2 * grayscaled[i][j+1]);
            gy += (1 * grayscaled[i+1][j+1]);
            
            grad[i][j][0] = gx;
            grad[i][j][1] = gy;
         }
      }
      return grad;
   }





    
    private static Color getColor(BufferedImage image, int x, int y) {
        int rgb = image.getRGB(x,y);
        return new Color(rgb);
    }
    
    private static void setColor(BufferedImage image, int x, int y, Color c) {
        int rgb = c.getRGB();
        image.setRGB(x,y,rgb);
    }

    /*
     *  None of the following methods were written by me, they are only used
     *  for ease of image input and output
     */

    private static void writeImage(BufferedImage image, String filename) {
        try {
            File outputFile = new File(filename);
            ImageIO.write(image, "png", outputFile);
        }
        catch(IOException io) {
            System.out.println("Encountered an error writing to file: ");
            System.out.println(io);
            System.exit(0);
        }
    }

    private static BufferedImage readImage(String filename) {
        try {
            return ImageIO.read(new File(filename));
        }
        catch(IOException io) {
            System.out.println("Encountered an error reading from file: ");
            System.out.println(io);
            System.exit(0);
            return null;
        }
    }
    
    private static BufferedImage deepCopy(BufferedImage bi) {
        ColorModel cm = bi.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(null);
        return new BufferedImage(cm, raster, isAlphaPremultiplied, null);
    }
}