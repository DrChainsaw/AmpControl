package ampcontrol.model.training.model;

import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * {@link ModelAdapter} for {@link ComputationGraph ComputationGraphs}.
 *
 * @author Christian Skärby
 */
public class GraphModelAdapter implements ModelAdapter {

    private static final Logger log = LoggerFactory.getLogger(GraphModelAdapter.class);

    private final ComputationGraph graph;

    public GraphModelAdapter(ComputationGraph graph) {
        this.graph = graph;
        for (GraphVertex l: graph.getVertices()) {

            if(l.getLayer() != null) {
                log.info(l.toString());
                log.info(l.getLayer().toString());
            } else {
                log.info(l.toString() + "inputs=" + Arrays.toString(l.getInputVertices()) + ",outputs=" + Arrays.toString(l.getOutputVertices()));
            }
        }
    }

    @Override
    public void fit(DataSetIterator iter) {
        graph.fit(iter);
    }

    @Override
    public <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals)  {
        return graph.doEvaluation(iter, evals);
    }

    @Override
    public Model asModel() {
        return graph;
    }
}
