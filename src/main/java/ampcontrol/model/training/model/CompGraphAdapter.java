package ampcontrol.model.training.model;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * {@link ModelAdapter} which wraps a computation graph.
 *
 * @author Christian Skärby
 */
public interface CompGraphAdapter extends ModelAdapter {

    @Override
    ComputationGraph asModel();
}
