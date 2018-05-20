package ampcontrol.model.training.listen;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.BiConsumer;

/**
 * {@link BaseTrainingListener} which evaluates the given {@link Model} and notifies a {@link BiConsumer} of the training
 * accuracy per iteration.
 * TODO: Too painful to test due to dependencies to Dl4j internals. Possible to redesign?
 *
 * @author Christian Skärby
 */
public class TrainEvaluator extends BaseTrainingListener {

    private static final Logger log = LoggerFactory.getLogger(TrainEvaluator.class);

    private final BiConsumer<Integer, Double> iterAndEvalListener;

    private int iterStore = 0;
    private final Evaluation eval;

    public TrainEvaluator(
            BiConsumer<Integer, Double> iterAnEvalListener) {
        this.iterAndEvalListener = iterAnEvalListener;
        this.eval = new Evaluation();
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {

        if (model instanceof MultiLayerNetwork) {
            final BaseOutputLayer ol = (BaseOutputLayer) ((MultiLayerNetwork) model).getOutputLayer();
            final INDArray labels = ol.getLabels();
            eval.eval(labels, ol.output(false));
            iterStore = iteration;

        } else if (model instanceof ComputationGraph) {
            final BaseOutputLayer ol = (BaseOutputLayer) ((ComputationGraph) model).getOutputLayer(0);
            final INDArray labels = ol.getLabels();
            eval.eval(labels, ol.output(false));
            iterStore = iteration;
        } else {
            throw new IllegalArgumentException("Not supported: " + model);
        }
    }

    @Override
    public void onEpochStart(Model model) {
        eval.reset();
    }

    @Override
    public void onEpochEnd(Model model) {
        log.info("Training accuracy at iteration " + iterStore + ": " + eval.accuracy());
        iterAndEvalListener.accept(iterStore, eval.accuracy());
    }
}
