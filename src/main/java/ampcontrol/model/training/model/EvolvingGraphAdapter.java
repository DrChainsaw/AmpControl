package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.Evolving;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.mutate.state.MutationState;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.function.Function;


/**
 * {@link ModelAdapter} which may evolve through a {@link Mutation}.
 *
 * @author Christian Skärby
 */
public class EvolvingGraphAdapter implements ModelAdapter, Evolving<EvolvingGraphAdapter> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingGraphAdapter.class);

    private final ComputationGraph graph;
    private final MutationState<ComputationGraphConfiguration.GraphBuilder> mutation;
    private final Function<ComputationGraph, ParameterTransfer> parameterTransferFactory;

    public EvolvingGraphAdapter(
            ComputationGraph graph,
            MutationState<ComputationGraphConfiguration.GraphBuilder> mutation) {
        this(graph, mutation, ParameterTransfer::new);
    }

    public EvolvingGraphAdapter(
            ComputationGraph graph,
            MutationState<ComputationGraphConfiguration.GraphBuilder> mutation,
            Function<ComputationGraph, ParameterTransfer> parameterTransferFactory) {
        this.graph = graph;
        this.mutation = mutation;
        this.parameterTransferFactory = parameterTransferFactory;
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

    @Override
    public void saveModel(String fileName) throws IOException {
        ModelAdapter.super.saveModel(fileName);
        mutation.save(fileName);
    }

    /**
     * Evolve the graph adapter
     *
     * @return the evolved adapter
     */
    @Override
    public EvolvingGraphAdapter evolve() {
        log.info("Evolve " + this);
        final MutationState<ComputationGraphConfiguration.GraphBuilder> newMutationState = mutation.clone();
        final ComputationGraphConfiguration.GraphBuilder mutated = newMutationState.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration().clone(),
                        new NeuralNetConfiguration.Builder(graph.conf().clone())));
        final ParameterTransfer parameterTransfer = parameterTransferFactory.apply(graph);

        final ComputationGraph newGraph = new ComputationGraph(mutated.build());
        newGraph.init();
        newGraph.getConfiguration().setIterationCount(graph.getIterationCount());
        newGraph.getConfiguration().setEpochCount(graph.getEpochCount());
        return new EvolvingGraphAdapter(parameterTransfer.transferWeightsTo(newGraph), newMutationState, parameterTransferFactory);
    }
}
