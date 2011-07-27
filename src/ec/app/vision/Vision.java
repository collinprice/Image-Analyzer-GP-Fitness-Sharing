package ec.app.vision;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import javax.imageio.ImageIO;

import ec.EvolutionState;
import ec.Individual;
import ec.app.vision.GPImage;
import ec.gp.GPIndividual;
import ec.gp.GPProblem;
import ec.simple.SimpleProblemForm;
import ec.util.Parameter;

/**********************************************************************
 * 
 * Collin Price
 * cp06vz @ brocku.ca
 * 3814647
 * 
 * COSC 4V82 Project
 * 
 * May., 9, 2011
 * 
 **********************************************************************/

public class Vision extends GPProblem implements SimpleProblemForm
    {
	private static final long serialVersionUID = 8886023049823102982L;
	
	public static GPImage trainingImage;
	public static GPImage testingImage;
	
	public int currentp1;
	public int currentp2;
	public int currentp3;
	public int currentp4;
	public int currentp5;
	public int currentp6;
	public int currentp7;
	public int currentp8;
	public int currentp9;
	public double stdDev;
	public double mean;
        
    public Object clone() {
        Vision newobj = (Vision) (super.clone());
        return newobj;
    }

    /**
     * Setup the GP for training.
     */
    public void setup(final EvolutionState state, final Parameter base) {
        super.setup(state,base);
        
        try {
    		trainingImage = new GPImage(ImageIO.read(state.parameters.getFile(base.push("training." + 0 + ".image"), null)), 
    										ImageIO.read(state.parameters.getFile(base.push("training." + 0 + ".image.mask"), null)));
    		testingImage = new GPImage(ImageIO.read(state.parameters.getFile(base.push("testing." + 0 + ".image"), null)), 
											ImageIO.read(state.parameters.getFile(base.push("testing." + 0 + ".image.mask"), null)));
		} catch (IOException e) {
			System.out.println("Cannot find an image...");
			e.printStackTrace();
			System.exit(1);
		}
		
    } // setup

    /**
     * Evaluation for each individual in the population.
     */
    public void evaluate(final EvolutionState state, final Individual ind, 
        final int subpopulation, final int threadnum) {
    	
    	int TP = 0;
    	int TN = 0;
    	SharedFitness f = (SharedFitness) ind.fitness;
        f.pixelFitness = new double[trainingImage.getWidth()][trainingImage.getHeight()];	
    	for (int x = 1; x < trainingImage.getWidth()-1; x++) {
    		for (int y = 1; y < trainingImage.getHeight()-1; y++) {
    		
    			
    			DoubleData input = new DoubleData();
    			
    			currentp1 = trainingImage.pixelData(x-1, y-1);
    			currentp2 = trainingImage.pixelData(x, y-1);
    			currentp3 = trainingImage.pixelData(x+1, y-1);
    			currentp4 = trainingImage.pixelData(x-1, y);
    			currentp5 = trainingImage.pixelData(x, y);
    			currentp6 = trainingImage.pixelData(x+1, y);
    			currentp7 = trainingImage.pixelData(x-1, y+1);
    			currentp8 = trainingImage.pixelData(x, y+1);
    			currentp9 = trainingImage.pixelData(x+1, y+1);
    			stdDev = trainingImage.getStdDev(x, y);
    			mean = trainingImage.getMean(x, y);
    			
    			((GPIndividual)ind).trees[0].child.eval(
                        state,threadnum,input,stack,((GPIndividual)ind),this);
    			
    			f.pixelFitness[x][y] = 0.0;
    			if (input.x > 0) {
    				if (trainingImage.maskData(x, y)) {
    					f.pixelFitness[x][y] = 1.0;
    					TP++;
    				}
    			} else {
    				if (!trainingImage.maskData(x, y)) {
    					f.pixelFitness[x][y] = 1.0;
    					TN++;
    				}
    			}
    		}
    	}
        
    	f.score = TP + TN;
    	
    } // evaluate
    
    public void closeContacts(EvolutionState state, int results) {
    	BufferedImage coded = null;
    	BufferedImage new_mask = null;
    	
    	SimpleStatistics stats = (SimpleStatistics) state.statistics;
    	Individual ind = stats.best_of_run[0];

    	int TP = 0;
    	int TN = 0;
    	int FP = 0;
    	int FN = 0;
    	
    	coded = new BufferedImage(testingImage.getWidth(), testingImage.getHeight(), BufferedImage.TYPE_INT_RGB);
    	new_mask = new BufferedImage(testingImage.getWidth(), testingImage.getHeight(), BufferedImage.TYPE_INT_RGB);
    	for (int y = 1; y < testingImage.getHeight()-1; y++) {
    		for (int x = 1; x < testingImage.getWidth()-1; x++) {
    		
    			DoubleData input = new DoubleData();
    			
    			currentp1 = testingImage.pixelData(x-1, y-1);
    			currentp2 = testingImage.pixelData(x, y-1);
    			currentp3 = testingImage.pixelData(x+1, y-1);
    			currentp4 = testingImage.pixelData(x-1, y);
    			currentp5 = testingImage.pixelData(x, y);
    			currentp6 = testingImage.pixelData(x+1, y);
    			currentp7 = testingImage.pixelData(x-1, y+1);
    			currentp8 = testingImage.pixelData(x, y+1);
    			currentp9 = testingImage.pixelData(x+1, y+1);
    			stdDev = testingImage.getStdDev(x, y);
    			mean = testingImage.getMean(x, y);
    			
    			((GPIndividual)ind).trees[0].child.eval(
                        state,0,input,stack,((GPIndividual)ind),this);
    			
    			if (input.x > 0) {
    				coded.setRGB(x, y, Color.red.getRGB());
    				if (testingImage.maskData(x, y)) {
    					new_mask.setRGB(x, y, Color.green.getRGB());
    					TP++;
    				} else {
    					new_mask.setRGB(x, y, Color.yellow.getRGB());
    					FP++;
    				}
    			} else {
    				coded.setRGB(x, y, new Color(testingImage.pixelData(x, y),testingImage.pixelData(x, y),testingImage.pixelData(x, y)).getRGB());
    				if (!testingImage.maskData(x, y)) {
    					new_mask.setRGB(x, y, Color.blue.getRGB());
    					TN++;
    				} else {
    					new_mask.setRGB(x, y, Color.red.getRGB());
    					FN++;
    				}
    			}
    		}
    	}
    	state.output.println("", stats.statisticslog);
    	state.output.println("Testing TP = " + TP + ", TN = " + TN ,stats.statisticslog);
    	state.output.println("Testing FP = " + FP + ", FN = " + FN ,stats.statisticslog);
    	short name = (short)System.currentTimeMillis();
    	
    	try {
			BufferedWriter out = new BufferedWriter(new FileWriter("" + name + "_stats.txt"));
			out.write("Testing TP = " + TP + ", TN = " + TN);
			out.newLine();
			out.write("Testing FP = " + FP + ", FN = " + FN);
			out.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
    	
    	try {
			ImageIO.write(coded, "png", new File("output" + name + "_masked" + ".png"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			ImageIO.write(new_mask, "png", new File("output" + name + ".png"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    } //closeContacts
    
} // Regression

