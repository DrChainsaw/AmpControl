package ampcontrol.model.training.model;

import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * {@link ModelAdapter} for {@link MultiLayerNetwork MultiLayerNetworks}.
 *
 * @author Christian Skärby
 */
public class MultiLayerModelAdapter implements ModelAdapter {

    private static final Logger log = LoggerFactory.getLogger(MultiLayerModelAdapter.class);

    private final MultiLayerNetwork model;

    public MultiLayerModelAdapter(MultiLayerNetwork model) {
        this.model = model;
        for (Layer l: model.getLayers()) {
            log.info(l.toString());
        }
    }

    @Override
    public void fit(DataSetIterator iter) {
        model.fit(iter);
    }

    @Override
    public <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals) {
        return model.doEvaluation(iter, evals);
    }

    @Override
    public Model asModel() {
        return model;
    }
}
