package ampControl.model.training.listen;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * {@link IterationListener} which evaluates the given {@link Model} and notifies a {@link BiConsumer} of the training
 * accuracy per iteration.
 * TODO: Too painful to test due to dependencies to Dl4j internals. Possible to redesign?
 * @author Christian Skärby
 */
public class TrainEvaluator implements TrainingListener {

    private static final Logger log = LoggerFactory.getLogger(TrainEvaluator.class);

    private final int resetAfterNumExamples;
    private final BiConsumer<Integer, Double> iterAndEvalListener;

    private int resetCount;
    private int iterStore = 0;
    private Evaluation eval;

    public TrainEvaluator(
            int resetAfterNumExamples,
            BiConsumer<Integer, Double> iterAnEvalListener) {
        this.resetAfterNumExamples = resetAfterNumExamples;
        this.iterAndEvalListener = iterAnEvalListener;
        this.resetCount = resetAfterNumExamples+1;
    }

    private boolean invoked = false;

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration) {

        if (model instanceof MultiLayerNetwork) {
            final BaseOutputLayer ol = (BaseOutputLayer) ((MultiLayerNetwork) model).getOutputLayer();
            final INDArray labels = ol.getLabels();
            checkEvalReset(labels.shape()[1]);
            resetCount += model.batchSize();
            eval.eval(labels, ol.output(false));
            iterStore = iteration;

        } else if (model instanceof ComputationGraph){
            final BaseOutputLayer ol = (BaseOutputLayer) ((ComputationGraph) model).getOutputLayer(0);
            final INDArray labels = ol.getLabels();
            checkEvalReset(labels.shape()[1]);
            resetCount += model.batchSize();
            eval.eval(labels, ol.output(false));
            iterStore = iteration;
        } else {
            throw new RuntimeException("Not supported: " + model);
        }
    }

    private void checkEvalReset(int nrofLabels) {
        if (resetCount > resetAfterNumExamples) {
            log.info("Reset training evaluator!");
            eval = new Evaluation( nrofLabels);
            resetCount = 0;
        }
    }

    @Override
    public void onEpochStart(Model model) {

    }

    @Override
    public void onEpochEnd(Model model) {
        if(eval != null) {
            log.info("Training accuracy after "+ resetCount + " examples: " + eval.accuracy());
            iterAndEvalListener.accept(iterStore, eval.accuracy());
        }
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {

    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {

    }

    @Override
    public void onGradientCalculation(Model model) {

    }

    @Override
    public void onBackwardPass(Model model) {

    }
}
