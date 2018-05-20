package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.IsNaN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * Notifies a callback {@link Runnable} if any activations are NaN
 *
 * @author Christian Skärby
 */
public class NanActivationWatcher extends BaseTrainingListener {

    private static final Logger log = LoggerFactory.getLogger(NanActivationWatcher.class);

    private final Condition isNan = new IsNaN();
    private final Runnable nanCallback;

    /**
     * Default constructor. Throws a {@link IllegalStateException} if any activations are NaN
     */
    public NanActivationWatcher() {
        this(() -> {
            throw new IllegalStateException();
        });
    }

    /**
     * Constructor
     *
     * @param nanCallback Callback in case of NaN activation
     */
    public NanActivationWatcher(Runnable nanCallback) {
        this.nanCallback = nanCallback;
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        for (int layer = 0; layer < activations.size(); layer++) {
            INDArray act = activations.get(layer);
            boolean nanExist = BooleanIndexing.or(act, isNan);
            if (nanExist) {
                if (log.isWarnEnabled()) {
                    log.warn("Layer: " + ((MultiLayerNetwork) model).getLayer(layer));
                    log.warn("Prev Layer: " + ((MultiLayerNetwork) model).getLayer(layer - 1));
                    // log.warn(BooleanIndexing.or(((MultiLayerNetwork)model).getLayer(layer).getParam("W"),isNan));
                    //log.warn(BooleanIndexing.or(((MultiLayerNetwork)model).getLayer(layer).getParam("b"),isNan));
                    log.warn("Nan activation from layer " + layer + " act: \n" + activations.get(layer) + "\nprev: \n" + activations.get(layer - 1));
                }
                nanCallback.run();
            }
        }
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        for (Map.Entry<String, INDArray> actEntry : activations.entrySet()) {
            INDArray act = actEntry.getValue();
            boolean nanExist = BooleanIndexing.or(act, isNan);
            if (nanExist) {
                if (log.isWarnEnabled()) {
                    final String layer = actEntry.getKey();
                    log.warn("Layer: " + ((ComputationGraph) model).getLayer(layer));
                    //log.warn(BooleanIndexing.or(((MultiLayerNetwork)model).getLayer(layer).getParam("W"),isNan));
                    //log.warn(BooleanIndexing.or(((MultiLayerNetwork)model).getLayer(layer).getParam("b"),isNan));
                    log.warn("Nan activation from layer " + layer + " act: \n" + activations.get(layer));
                }
                nanCallback.run();
            }
        }
    }
}
