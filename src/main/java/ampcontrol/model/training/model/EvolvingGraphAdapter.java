package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.CrossBreeding;
import ampcontrol.model.training.model.evolve.Evolving;
import ampcontrol.model.training.model.evolve.crossover.graph.GraphInfo;
import ampcontrol.model.training.model.evolve.crossover.state.CrossoverState;
import ampcontrol.model.training.model.evolve.crossover.state.NoStateWapper;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.mutate.state.MutationState;
import ampcontrol.model.training.model.evolve.mutate.state.NoMutationStateWapper;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;


/**
 * {@link ModelAdapter} which may evolve through a {@link Mutation}.
 *
 * @author Christian Skärby
 */
public class EvolvingGraphAdapter<S> implements CompGraphAdapter, Evolving<EvolvingGraphAdapter<S>>, CrossBreeding<EvolvingGraphAdapter<S>> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingGraphAdapter.class);

    private final ComputationGraph graph;
    private final MutationState<ComputationGraphConfiguration.GraphBuilder> mutation;
    private final CrossoverState<GraphInfo, S> crossover;
    private final BiFunction<
            Function<String, GraphVertex>,
            Function<GraphVertex, ComputationGraph>,
            ParameterTransfer> parameterTransferFactory;

    /**
     * Create a builder for this class
     *
     * @param graph {@link ComputationGraph} to wrap
     * @return a new Builder
     */
    public static <S> Builder<S> builder(ComputationGraph graph) {
        return new Builder<>(graph);
    }

    // Use builder to create
    private EvolvingGraphAdapter(
            ComputationGraph graph,
            MutationState<ComputationGraphConfiguration.GraphBuilder> mutation,
            CrossoverState<GraphInfo, S> crossover,
            BiFunction<
                    Function<String, GraphVertex>,
                    Function<GraphVertex, ComputationGraph>,
                    ParameterTransfer> parameterTransferFactory) {
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
    public EvolvingGraphAdapter<S> evolve() {
        log.info("Evolve " + this);
        final MutationState<ComputationGraphConfiguration.GraphBuilder> newMutationState = mutation.clone();
        final CrossoverState<GraphInfo, S> newCrossoverState = crossover.clone();
        final ComputationGraphConfiguration.GraphBuilder mutated = newMutationState.mutate(
                CompGraphUtil.toBuilder(graph));
        final ParameterTransfer parameterTransfer = parameterTransferFactory.apply(graph::getVertex, gv -> graph);

        return createOffspring(newMutationState, newCrossoverState, mutated, parameterTransfer);
    }

    @Override
    public EvolvingGraphAdapter<S> cross(EvolvingGraphAdapter<S> mate) {
        final MutationState<ComputationGraphConfiguration.GraphBuilder> newMutationState = mutation.clone();
        final CrossoverState<GraphInfo, S> newCrossoverState = crossover.clone();

        final GraphInfo thisGraph = new GraphInfo.Input(CompGraphUtil.toBuilder(graph));
        final GraphInfo otherGraph = new GraphInfo.Input(CompGraphUtil.toBuilder(mate.graph));
        final GraphInfo result = newCrossoverState.cross(thisGraph, otherGraph);

        newCrossoverState.merge(mate.crossover, thisGraph, otherGraph, result);

        final Map<String, GraphVertex> nameToVertex =
                Stream.concat(
                        result.verticesFrom(thisGraph)
                                .map(nameMapping -> new AbstractMap.SimpleEntry<>(nameMapping.getNewName(), graph.getVertex(nameMapping.getOldName()))),
                        result.verticesFrom(otherGraph)
                                .map(nameMapping -> new AbstractMap.SimpleEntry<>(nameMapping.getNewName(), mate.graph.getVertex(nameMapping.getOldName()))))
                        .collect(Collectors.toMap(
                                Map.Entry::getKey,
                                Map.Entry::getValue
                        ));

        final Function<String, GraphVertex> nameToVertexFunction = nameToVertex::get;
        // Should perhaps be done in a more rigid manner...
        final Function<GraphVertex, ComputationGraph> vertexToGraph = graphVertex ->
                Optional.ofNullable(graph.getVertex(graphVertex.getVertexName())).isPresent() ? graph : mate.graph;

        final ParameterTransfer parameterTransfer = parameterTransferFactory.apply(nameToVertexFunction, vertexToGraph);
        return createOffspring(
                newMutationState,
                newCrossoverState,
                result.builder(),
                parameterTransfer);
    }

    @NotNull
    private EvolvingGraphAdapter<S> createOffspring(
            MutationState<ComputationGraphConfiguration.GraphBuilder> newMutationState,
            CrossoverState<GraphInfo, S> newCrossoverState,
            ComputationGraphConfiguration.GraphBuilder mutated,
            ParameterTransfer parameterTransfer) {
        final ComputationGraph newGraph = new ComputationGraph(mutated.build());
        newGraph.init();
        newGraph.getConfiguration().setIterationCount(graph.getIterationCount());
        newGraph.getConfiguration().setEpochCount(graph.getEpochCount());
        return new EvolvingGraphAdapter<>(parameterTransfer.transferWeightsTo(newGraph), newMutationState, newCrossoverState, parameterTransferFactory);
    }

    public static class Builder<S> {
        private final ComputationGraph graph;
        private MutationState<ComputationGraphConfiguration.GraphBuilder> mutation = new NoMutationStateWapper<>(builder -> builder);
        private CrossoverState<GraphInfo, S> crossover = new NoStateWapper<>((info1, info2) -> info1);
        private BiFunction<
                Function<String, GraphVertex>,
                Function<GraphVertex, ComputationGraph>,
                ParameterTransfer> parameterTransferFactory = (graph, gvToCg) -> new ParameterTransfer(graph);

        private Builder(ComputationGraph graph) {
            this.graph = graph;
        }

        /**
         * Sets the mutation to use when evolving
         *
         * @param mutation mutation to use
         * @return the Builder for fluent API
         */
        public Builder<S> mutation(MutationState<ComputationGraphConfiguration.GraphBuilder> mutation) {
            this.mutation = mutation;
            return this;
        }

        /**
         * Sets the crossover to use when crossbreeding
         *
         * @param crossover crossover to use
         * @return the Builder for fluent API
         */
        public Builder<S> crossover(CrossoverState<GraphInfo, S> crossover) {
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
        public Builder<S> paramTransfer(BiFunction<
                Function<String, GraphVertex>,
                Function<GraphVertex, ComputationGraph>,
                ParameterTransfer> parameterTransferFactory) {
            this.parameterTransferFactory = parameterTransferFactory;
            return this;
        }

        public EvolvingGraphAdapter<S> build() {
            return new EvolvingGraphAdapter<>(graph, mutation, crossover, parameterTransferFactory);
        }
    }
}
