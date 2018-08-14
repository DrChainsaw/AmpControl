package ampcontrol.model.training.listen;

import ampcontrol.model.visualize.Plot;
import ampcontrol.model.visualize.RealTimePlot;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

public class SeBlockInspection extends BaseTrainingListener {

    private final Pattern actPattern = Pattern.compile("excite\\d*");
    private final Plot.Factory<Integer, Double> plotFactory = title -> new RealTimePlot<>(title, "");
    private final Map<String, ActivationListener> plots = new HashMap<>();
    private final Map<String, double[][]> activations = new HashMap<>();

    private final static class ActivationListener {

        private final int maxNrofSeries;
        private final Plot<Integer, Double> plot;
        private int currSeries = 0;

        private ActivationListener(int maxNrofSeries, Plot<Integer, Double> plot) {
            this.maxNrofSeries = maxNrofSeries;
            this.plot = plot;
        }

        private void plotActivation(double[] valsToPlot) {
            final String series = String.valueOf(currSeries++ % maxNrofSeries);
            for (int i = 0; i < valsToPlot.length; i++) {
                plot.plotData(series, i, valsToPlot[i]);
            }
        }
    }


    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        this.activations.clear();
        for (Map.Entry<String, INDArray> actEntry : activations.entrySet()) {
            final String vertName = actEntry.getKey();
            if (actPattern.matcher(vertName).matches()) {
                this.activations.put(vertName, actEntry.getValue().toDoubleMatrix());
            }
        }
    }

    @Override
    public void onBackwardPass(Model model) {
        if (model instanceof ComputationGraph) {
            final BaseOutputLayer ol = (BaseOutputLayer) ((ComputationGraph) model).getOutputLayer(0);
            final INDArray labelInds = ol.getLabels().argMax(1);

            plotActivationsPerLabel(labelInds);
        } else {
            throw new UnsupportedOperationException("Not implemented!");
        }
    }

    private void plotActivationsPerLabel(INDArray labelInds) {
        for (Map.Entry<String, double[][]> actEntry : this.activations.entrySet()) {

            final double[][] acts = actEntry.getValue();
            for(int i = 0; i < acts.length; i++) {
                final int labelInd = labelInds.getInt(i);
                final String plotName = actEntry.getKey()+ "_labelInd_" + labelInd;
                final ActivationListener listener = plots.computeIfAbsent(plotName, name -> new ActivationListener(30, plotFactory.create(name)));
                listener.plotActivation(acts[i]);
            }
        }
    }
}
