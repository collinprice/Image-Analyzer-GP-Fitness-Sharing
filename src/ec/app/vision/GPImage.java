package ec.app.vision;

import java.awt.Color;
import java.awt.image.BufferedImage;

/**
 * This class is used to store image data. It stores both the black and white data, and the masking data.
 */

public class GPImage {

	private int[][]		srcData;
	private boolean[][]	maskData;
	private double[][]	stdDev;
	private double[][]	meanVals;
	
	private int neg;
	private int pos;
	private int width;
	private int height;
	
	public GPImage(BufferedImage src, BufferedImage mask) {
		srcData = new int[src.getWidth()][src.getHeight()];
		maskData = new boolean[mask.getWidth()][mask.getHeight()];
		width = src.getWidth();
		height = src.getHeight();
		
		for (int y = 0; y < src.getHeight(); y++) {
			for (int x = 0; x < src.getWidth(); x++) {
				Color p = new Color(src.getRGB(x, y));
				srcData[x][y] = p.getRed();
			}
		}
		
		for (int y = 0; y < mask.getHeight(); y++) {
			for (int x = 0; x < mask.getWidth(); x++) {
				Color p = new Color(mask.getRGB(x, y));
				if (p.getRed() == p.getGreen() && p.getRed() == p.getBlue()) {
					maskData[x][y] = false;
				} else {
					maskData[x][y] = true;
				}
			}
		}
		
		calcStdDev(srcData);
	} // constructor
	
	public int totalPositives() {
		return pos;
	} // totalPositives
	
	public int totalNegatives() {
		return neg;
	} // totalNegatives
	
	/*
	 * Returns true if the pixel is an airplane
	 */
	public boolean maskData(int x, int y) {
		return maskData[x][y];
	} // maskData
	
	/*
	 * Returns the black and white pixel value.
	 */
	public int pixelData(int x, int y) {
		return srcData[x][y];
	} // pixelData
	
	public double getStdDev(int x, int y) {
		return stdDev[x][y];
	} // getStdDev
	
	public double getMean(int x, int y) {
		return meanVals[x][y];
	} // getMean
	
	public int getHeight() {
		return height;
	} // getHeight
	
	public int getWidth() {
		return width;
	} // getWidth
	
	/*
     * Calculates the Standard Deviation and Mean of each pixel.
     */
    private void calcStdDev(int[][] pop) {
    	stdDev = new double[pop.length][pop[0].length];
    	meanVals = new double[pop.length][pop[0].length];
    	
    	for (int i = 1; i < pop.length-1; i++) {
    		for (int j = 1; j < pop[0].length-1; j++) {
    			int total = 0;
    			
    			int[] subset = {
    					pop[i-1][j-1],
    					pop[i][j-1],
    					pop[i+1][j-1],
    					pop[i-1][j],
    					pop[i][j],
    					pop[i+1][j],
    					pop[i-1][j+1],
    					pop[i][j+1],
    					pop[i+1][j+1]
    					         };
    			
    			for (int x : subset) {
    				total += x;
    			}
    			
    			int mean = total / subset.length;
    			
    			total = 0;
    			for (int x : subset) {
    				total += x - mean;
    			}
    			
    			stdDev[i][j] = Math.sqrt(total / subset.length);
    			meanVals[i][j] = mean;
    		}
    	}
    } // calcStdDev
	
} // GPImage
