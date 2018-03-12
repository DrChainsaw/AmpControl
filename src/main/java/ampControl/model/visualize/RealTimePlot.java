
package ampControl.model.visualize;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.Styler.ChartTheme;

/**
 * Real time updatable plot with support for an arbitrary number of series. Can also serialize the plotted data and
 * recreate a plot from such data. Typically used for plot training/eval metrics for each iteration. Note: The amount
 * of data points per timeseries is limited to 1000 as a significant slowdown was observed for higher numbers. When 1000
 * points is reached, all even points will be removed. New points after this will be added as normal until the total hits
 * 1000 again.
 *
 * @author Christian Skärby
 */
public class RealTimePlot<X extends Number, Y extends Number> {

    private final String title;
    private final XYChart xyChart;
    private final SwingWrapper<XYChart> swingWrapper;
    private final String plotDir;

    private final Map<String, DataXY<X,Y>> plotSeries = new HashMap<>();

    private static class DataXY<X extends Number, Y extends Number> implements Serializable {

        private static final long serialVersionUID = 7526471155622776891L;
        DataXY(String series) {this.series = series;}

        private final String series;

        private final LinkedList<X> xData = new LinkedList<>();
        private final LinkedList<Y> yData = new LinkedList<>();

        private void addPoint(X x, Y y, XYChart xyChart, SwingWrapper<XYChart> swingWrapper) {
            xData.addLast(x);
            yData.addLast(y);
            if (xData.size() > 1000) {
                for (int i = 0; i < xData.size(); i += 2) {
                    xData.remove(i);
                    yData.remove(i);
                }

            }


            if (!xyChart.getSeriesMap().containsKey(series)) {
                xyChart.addSeries(series, xData, yData, null);
            } else {
                xyChart.updateXYSeries(series, xData, yData, null);
                javax.swing.SwingUtilities.invokeLater(swingWrapper::repaintChart);
            }
        }

        public void createSeries(XYChart xyChart, SwingWrapper<XYChart> swingWrapper) {
            if(xData.size() == 0) {
                xyChart.addSeries(series, Arrays.asList(0), Arrays.asList(1));
            } else {
                xyChart.addSeries(series, xData, yData);
            }
            try {
                javax.swing.SwingUtilities.invokeAndWait(swingWrapper::repaintChart);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Constructor
     * @param title Title of the plot
     * @param plotDir Directory to store plots in.
     */
    public RealTimePlot(String title, String plotDir) {
        // Create Chart
        this.title = title;
        xyChart = new XYChartBuilder().width(1000).height(500).theme(ChartTheme.Matlab).title(title).build();
        xyChart.getStyler().setLegendPosition(Styler.LegendPosition.OutsideE);
        xyChart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);

        this.swingWrapper = new SwingWrapper<>(xyChart);
        swingWrapper.displayChart();
        this.plotDir = plotDir;
    }

    /**
     * Plot some data belonging to a certain label. Will be appended to an existing timeseries of such exists, either in
     * an existing window or in serialized format in the plotDir. If no timeseries with the given label exists it will
     * be created in the window of this plot instance.
     * @param label time series label
     * @param x point on x axis
     * @param y point on y axis
     */
    public void plotData(String label, X x, Y y) {
        // new Thread(() -> {
        DataXY<X,Y> data = plotSeries.get(label);
        if (data == null) {
            data = createSeries(label);
        }
        data.addPoint(x, y, xyChart, swingWrapper);
        //  }).start();
    }

    /**
     * Creates a time series for the given label. If data with the given label exists in serialized format in the
     * plotDir the time series of that data will be recreated.
     * @param label
     * @return
     */
    public DataXY<X,Y> createSeries(String label) {
        DataXY<X,Y> data = plotSeries.get(label);
        if (data == null) {
            data = restoreOrCreatePlotData(label);
            plotSeries.put(label, data);
            data.createSeries(xyChart, swingWrapper);
        }
        return data;
    }

    /**
     * Serialize the data for the given label into a file in the plotDir.
     * @param label
     * @throws IOException
     */
    public void storePlotData(String label) throws IOException {
            DataXY<X,Y> data = plotSeries.get(label);
            if(data != null) {
                OutputStream file = new FileOutputStream(createFileName(label));
                OutputStream buffer = new BufferedOutputStream(file);
                ObjectOutput output = new ObjectOutputStream(buffer);
                output.writeObject(data);
                output.close();
                buffer.close();
                file.close();
            }
    }

    private DataXY<X,Y> restoreOrCreatePlotData(String label) {
        File dataFile = new File(createFileName(label));
        if(dataFile.exists()) {
            try {
            InputStream file = new FileInputStream(dataFile);
            InputStream buffer = new BufferedInputStream(file);
            ObjectInput input = new ObjectInputStream(buffer);
            return (DataXY<X,Y>) input.readObject();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }
        }
        return new DataXY<>(label);
    }

    private String createFileName(String label) {
        return plotDir + File.separator + title + "_" + label + ".plt";
    }

    public static void main(String[] args) {

        final RealTimePlot<Integer, Double> plotter = new RealTimePlot<>("Test plot", "");
        IntStream.range(1000, 2000).forEach(x -> Stream.of("s1", "s2", "s3").forEach(str -> {
            plotter.createSeries(str);
            plotter.plotData(str, x, 1d / ((double) x + 10));
        }));
//          try {
//            plotter.storePlotData("s1");
//          } catch (IOException e) {
//             e.printStackTrace();
//          }

    }
}