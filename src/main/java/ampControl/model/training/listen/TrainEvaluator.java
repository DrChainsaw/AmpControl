package ampControl.model.training.listen;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.function.BiConsumer;

/**
 * {@link IterationListener} which evaluates the given {@link Model} and notifies a {@link BiConsumer} of the training
 * accuracy per iteration.
 * TODO: Too painful to test due to dependencies to Dl4j internals. Possible to redesign?
 * @author Christian Skärby
 */
public class TrainEvaluator extends BaseTrainingListener {

    private final int resetAfterNumExamples;
    private final BiConsumer<Integer, Double> iterAndEvalListener;
    private static final Logger log = LoggerFactory.getLogger(TrainEvaluator.class);

    private int resetCount;
    private int labelsCount = 0;
    private int iterStore = 0;
    private List<INDArray> labels;
    private Evaluation eval;

    public TrainEvaluator(int resetAfterNumExamples,BiConsumer<Integer, Double> iterAnEvalListener) {
        this.resetAfterNumExamples = resetAfterNumExamples;
        this.iterAndEvalListener = iterAnEvalListener;
        resetCount = resetAfterNumExamples+1;
    }

    private boolean invoked = false;

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {

        if (model instanceof MultiLayerNetwork) {
            BaseOutputLayer ol = (BaseOutputLayer) ((MultiLayerNetwork) model).getOutputLayer();
            checkEvalReset();
            resetCount += labels.get(labelsCount).shape()[0];
            eval.eval(labels.get(labelsCount),output(ol));
            labelsCount++;
            iterStore = iteration;

        } else if (model instanceof ComputationGraph){
            BaseOutputLayer ol = (BaseOutputLayer) ((ComputationGraph) model).getOutputLayer(0);
            checkEvalReset();
            resetCount += labels.get(labelsCount).shape()[0];
            eval.eval(labels.get(labelsCount), output(ol));
            labelsCount++;
            iterStore = iteration;
        } else {
            throw new RuntimeException("Not supported: " + model);
        }
    }

    private void checkEvalReset() {
        if (resetCount > resetAfterNumExamples) {
            log.info("Reset training evaluator!");
            eval = new Evaluation( labels.get(0).shape()[1]);
            resetCount = 0;
        }
    }

    public void setLabels(List<INDArray> labels) {
        labelsCount = 0;
        this.labels = labels;
    }

    public void pollListener() {
        if(eval != null) {
            log.info("Training accuracy after "+ resetCount + " examples: " + eval.accuracy());
            iterAndEvalListener.accept(iterStore, eval.accuracy());
        }
    }

    private INDArray output(BaseOutputLayer outputLayer) {
        return outputLayer.layerConf().getActivationFn().getActivation(outputLayer.getPreOutput(), false);
    }
}
