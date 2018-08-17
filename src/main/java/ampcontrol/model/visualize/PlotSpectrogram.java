package ampcontrol.model.visualize;

import org.jzy3d.analysis.AbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.AWTChartComponentFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Utility class for plotting spectrograms and other 2D representations of sound.
 *
 * @author Christian Skärby
 */
public class PlotSpectrogram {
	
	private static class Analysis extends AbstractAnalysis {
		
		private final Mapper mapper;
		private final Range rangeFreq;
		private final Range rangeTime;
				
		public Analysis(Mapper mapper, Range rangeFreq, Range rangeTime) {
			super();
			this.mapper = mapper;
			this.rangeFreq = rangeFreq;
			this.rangeTime = rangeTime;
		}



		@Override
		public void init() throws Exception {

		        // Define range and precision for the function to plot
		        final int stepsFreq = (int)rangeFreq.getMax()+1;
		        final int stepsTime = (int)rangeTime.getMax()+1;
		        // Create the object to represent the function over the given range.
		        
		        final Shape surface = Builder.buildOrthonormal(new OrthonormalGrid(rangeFreq, stepsFreq, rangeTime, stepsTime), mapper);
		        surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new Color(1, 1, 1, .5f)));
		        surface.setFaceDisplayed(true);
		        surface.setWireframeDisplayed(false);

		        // Create a chart
		        chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType());
		        chart.getScene().getGraph().add(surface);	
		}
	}
	
	public static void plot(final INDArray specgram, int timeInd, int freqInd) {
		final long[] shape = specgram.shape();
		System.out.println("shape: " + Arrays.toString(shape));
		final Range rangeTime = new Range(0, shape[timeInd]-1);
		final Range rangeFreq = new Range(0, shape[freqInd]-1);

		final Mapper mapper = new Mapper() {
			
			@Override
			public double f(double freq, double time) {
				for(int i = 0; i < shape.length; i++) {
					shape[i] = 0;
					if(i == timeInd) {
						shape[i] = (int)time;
					}
					if(i == freqInd) {
						shape[i] = (int)freq;
					}
				}
				return specgram.getDouble(shape);
			}
		};
		try {
			AnalysisLauncher.open(new Analysis(mapper, rangeFreq, rangeTime));
		} catch (Exception e) {
			//System.out.println("plot failed! data: " + specgram);
			e.printStackTrace();
		}
	}
	
	public static void plot(double[][] specgram) {

		final Range rangeTime = new Range(0, specgram.length-1);
		final Range rangeFreq = new Range(0, specgram[0].length-1);

		final Mapper mapper = new Mapper() {
			
			@Override
			public double f(double freq, double time) {
				return specgram[(int)time][(int)freq];
			}
		};
		try {
			AnalysisLauncher.open(new Analysis(mapper, rangeFreq, rangeTime));
		} catch (Exception e) {
			//System.out.println("plot failed! data: " + specgram);
			e.printStackTrace();
		}
	}

}
