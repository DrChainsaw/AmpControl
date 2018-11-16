package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.Evolving;
import ampcontrol.model.training.model.evolve.crossover.Crossover;
import ampcontrol.model.training.model.evolve.crossover.graph.GraphInfo;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.mutate.state.MutationState;
import ampcontrol.model.training.model.evolve.mutate.state.NoMutationStateWapper;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.eval.IEvaluation;
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
public class EvolvingGraphAdapter implements CompGraphAdapter, Evolving<EvolvingGraphAdapter> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingGraphAdapter.class);

    private final ComputationGraph graph;
    private final MutationState<ComputationGraphConfiguration.GraphBuilder> mutation;
    private final Crossover<GraphInfo> crossover;
    private final Function<Function<String, ComputationGraph>, ParameterTransfer> parameterTransferFactory;

    /**
     * Create a builder for this class
     * @param graph {@link ComputationGraph} to wrap
     * @return a new Builder
     */
    public static Builder builder(ComputationGraph graph) {
        return new Builder(graph);
    }

    // Use builder to create
    private EvolvingGraphAdapter(
            ComputationGraph graph,
            MutationState<ComputationGraphConfiguration.GraphBuilder> mutation,
            Crossover<GraphInfo> crossover,
            Function<Function<String, ComputationGraph>, ParameterTransfer> parameterTransferFactory) {
        this.graph = graph;
        this.mutation = mutation;
        this.crossover = crossover;
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
    public ComputationGraph asModel() {
        return graph;
    }

    @Override
    public void saveModel(String fileName) throws IOException {
        CompGraphAdapter.super.saveModel(fileName);
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
        final ParameterTransfer parameterTransfer = parameterTransferFactory.apply(str -> graph);

        final ComputationGraph newGraph = new ComputationGraph(mutated.build());
        newGraph.init();
        newGraph.getConfiguration().setIterationCount(graph.getIterationCount());
        newGraph.getConfiguration().setEpochCount(graph.getEpochCount());
        return new EvolvingGraphAdapter(parameterTransfer.transferWeightsTo(newGraph), newMutationState, crossover, parameterTransferFactory);
    }

    public static class Builder {
        private final ComputationGraph graph;
        private MutationState<ComputationGraphConfiguration.GraphBuilder> mutation = new NoMutationStateWapper<>(builder -> builder);
        private Crossover<GraphInfo> crossover = (info1, info2) -> info1;
        private Function<Function<String, ComputationGraph>, ParameterTransfer> parameterTransferFactory = ParameterTransfer::new;

        private Builder(ComputationGraph graph) {
            this.graph = graph;
        }

        /**
         * Sets the mutation to use when evolving
         *
         * @param mutation mutation to use
         * @return the Builder for fluent API
         */
        public Builder mutation(MutationState<ComputationGraphConfiguration.GraphBuilder> mutation) {
            this.mutation = mutation;
            return this;
        }

        /**
         * Sets the crossover to use when crossbreeding
         *
         * @param crossover crossover to use
         * @return the Builder for fluent API
         */
        public Builder crossover(Crossover<GraphInfo> crossover) {
            this.crossover = crossover;
            return this;
        }

        /**
         * Sets the factory for {@link ParameterTransfer} which detemines how parameters shall be transferred from old
         * graphs to new when evolving or crossbreeding
         *
         * @param parameterTransferFactory Factory for {@link ParameterTransfer}
         * @return the Builder for fluent API
         */
        public Builder paramTransfer(Function<Function<String, ComputationGraph>, ParameterTransfer> parameterTransferFactory) {
            this.parameterTransferFactory = parameterTransferFactory;
            return this;
        }

        public EvolvingGraphAdapter build() {
            return new EvolvingGraphAdapter(graph, mutation, crossover, parameterTransferFactory);
        }
    }
}
