package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.Evolving;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * {@link ModelAdapter} which may evolve through a {@link Mutation}.
 *
 * @author Christian Skärby
 */
public class EvolvingGraphAdapter implements ModelAdapter, Evolving<EvolvingGraphAdapter> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingGraphAdapter.class);

    private final ComputationGraph graph;
    private final Mutation mutation;

    public EvolvingGraphAdapter(ComputationGraph graph, Mutation mutation) {
        this.graph = graph;
        this.mutation = mutation;
    }

    @Override
    public void fit(DataSetIterator iter) {
        graph.fit(iter);
    }

    @Override
    public <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals) {
        return graph.doEvaluation(iter, evals);
    }

    @Override
    public Model asModel() {
        return graph;
    }

    /**
     * Evolve the graph adapter
     *
     * @return the evolved adapter
     */
    @Override
    public synchronized EvolvingGraphAdapter evolve() {
        log.info("Evolve " + this);
        graph.getListeners().clear();
        final TransferLearning.GraphBuilder mutated = mutation.mutate(new TransferLearning.GraphBuilder(graph), graph);
        final ParameterTransfer parameterTransfer = new ParameterTransfer(graph);
        final ComputationGraph newGraph = mutated.build();
        newGraph.init();
        newGraph.getConfiguration().setIterationCount(graph.getIterationCount());
        newGraph.getConfiguration().setEpochCount(graph.getEpochCount());
        return new EvolvingGraphAdapter(parameterTransfer.transferWeightsTo(newGraph), mutation);
    }
}
